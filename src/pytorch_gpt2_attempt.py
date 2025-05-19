import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime
import math
import sys
import time

from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

from contextlib import nullcontext
from typing import List

from config import TrafoConfig
from data_utils import load_dataset, load_tokenizer, dataset_batch_iter, NBBatchQueue
from my_globals import CHART_DATA, CFG

class MHSA(nn.Module):
    def __init__(self, cfg: TrafoConfig):
        super(MHSA, self).__init__()
        self.cfg = cfg

        self.w_qkv = nn.Linear(self.cfg.embed_dim, 3 * self.cfg.embed_dim, bias=self.cfg.qkv_bias)
        self.w_atn_out = nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim, bias=True)

        nn.init.normal_(self.w_atn_out.weight, mean=0.0, std=0.02/math.sqrt(2 * self.cfg.num_blocks))

    def forward(self, xs: Tensor) -> Tensor:
        x_qkv: Tensor = self.w_qkv(xs)

        x_qkv_split: List[Tensor] = x_qkv.split(self.cfg.embed_dim, dim=-1)

        x_q, x_k, x_v = [x_single.view(-1, self.cfg.ctx_size, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2) for x_single in x_qkv_split]

        x_atn = F.scaled_dot_product_attention(x_q, x_k, x_v, attn_mask=None, is_causal=self.cfg.is_causal)

        x_out = self.w_atn_out(x_atn.transpose(1, 2).view(xs.size()))

        return x_out

class ModifiedMHSA(nn.Module):
    def __init__(self, cfg: TrafoConfig):
        super(ModifiedMHSA, self).__init__()
        self.cfg = cfg
        self.w_v = nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim, bias=self.cfg.qkv_bias)
        self.w_atn_out = nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim, bias=True)

        nn.init.normal_(self.w_atn_out.weight, mean=0.0, std=0.02/math.sqrt(2 * self.cfg.num_blocks))

    def forward(self, xs: Tensor) -> Tensor:
        x_v: Tensor = self.w_v(xs).view(-1, self.cfg.ctx_size, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2)

        x_reshaped = xs.view(-1, self.cfg.ctx_size, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2)

        x_atn = F.scaled_dot_product_attention(x_reshaped, x_reshaped, x_v, attn_mask=None, is_causal=self.cfg.is_causal)

        x_out = self.w_atn_out(x_atn.transpose(1, 2).view(xs.size()))

        return x_out
        
        
class TrafoBlock(nn.Module):
    def __init__(self, cfg: TrafoConfig):
        super(TrafoBlock, self).__init__()
        self.cfg = cfg

        self.ln_atn = nn.LayerNorm(self.cfg.embed_dim)

        # self.atn = MHSA(cfg)
        self.atn = ModifiedMHSA(cfg)

        self.ln_ff = nn.LayerNorm(self.cfg.embed_dim)
        
        self.ff1 = nn.Linear(self.cfg.embed_dim, self.cfg.ff_dim)
        self.ff2 = nn.Linear(self.cfg.ff_dim, self.cfg.embed_dim)

        nn.init.normal_(self.ff2.weight, mean=0.0, std=0.02/math.sqrt(2 * self.cfg.num_blocks))

        
    def forward(self, xs):
        x_ln_atn = self.ln_atn(xs)
        x_atn = self.atn(x_ln_atn)

        x_ln_ff = self.ln_ff(x_atn)
        x_ff1 = F.gelu(self.ff1(x_ln_ff))
        x_ff2 = self.ff2(x_ff1)

        return xs + x_ff2

        
class Trafo(nn.Module):
    def __init__(self, cfg: TrafoConfig):
        super(Trafo, self).__init__()
        self.cfg = cfg

        self.unembed = nn.Linear(self.cfg.embed_dim, self.cfg.vocab_size, bias=False)
        self.tok_embed = nn.Embedding(self.cfg.vocab_size, self.cfg.embed_dim, _weight=self.unembed.weight)
        self.pos_embed = nn.Embedding(self.cfg.ctx_size, self.cfg.embed_dim)

        self.blocks = nn.Sequential(*[TrafoBlock(cfg) for _ in range(self.cfg.num_blocks)])

        self.ln_unembed = nn.LayerNorm(self.cfg.embed_dim)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('w_atn_out.weight') or pn.endswith('ff2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * cfg.num_blocks))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, xs: Tensor, y_gt: Tensor|None = None):
        x_tok_embed = self.tok_embed(xs)

        allpos = torch.arange(0, self.cfg.ctx_size, dtype=torch.long, device=xs.device)
        pos_embeddings = self.pos_embed(allpos)

        x_pos_embed = x_tok_embed + pos_embeddings

        x_blocks = self.blocks(x_pos_embed)

        x_ln_unembed = self.ln_unembed(x_blocks)

        logits = self.unembed(x_ln_unembed)

        other = None
        
        if y_gt is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_gt.reshape(-1), ignore_index=-1)

            acc_scores = logits.argmax(-1) == y_gt

            acc = acc_scores.mean(dtype=torch.float)

            acc_hit_ids = logits.argmax(-1)[acc_scores]

            other = (loss, acc, acc_hit_ids)

        return logits, other

    @torch.no_grad()
    def generate(self, encoded_prompt: List[int], n_new_tokens: int, separator: int, topk: int, temperature: float, device):
        total_gen_len = len(encoded_prompt) + n_new_tokens
        assert total_gen_len <= self.cfg.ctx_size, f"total desired length must not exceed context size ({total_gen_len}, {self.cfg.ctx_size}, respectively)"

        tokens = encoded_prompt

        tok_tensor = torch.full((self.cfg.ctx_size,), separator, dtype=torch.long, device=device)

        for i in range(n_new_tokens):

            for tok_i, tok in enumerate(tokens):
                tok_tensor[tok_i] = tok

            logits, _ = self(tok_tensor.unsqueeze(0))

            last_logit: Tensor = logits[0, len(tokens) - 1]

            topk_values, topk_indices = last_logit.topk(topk)

            tok_idx = F.softmax(topk_values, dim=-1).multinomial(1)

            tok_id = topk_indices[tok_idx]

            tokens += tok_id.tolist()

        return tokens
         

def t_v_step(model, scaler, ctx, device, batch, ddp=False, master_process=True, in_eval=False):
    torch.set_grad_enabled(not in_eval)
    if in_eval:
        model.eval()

    batch_loss = torch.zeros((), device=device)
    batch_acc = torch.zeros((), device=device)

    batch_acc_hit_ids = set()

    mbatch_iter = tqdm(batch, unit="mbatches") if master_process and len(batch) >= 40 else batch

    for idx, mbatch in enumerate(mbatch_iter):
        if ddp:
            model.require_backward_grad_sync = idx == len(batch) - 1

        mbatch_tensor = torch.tensor(mbatch, dtype=torch.long, device=device)

        x = mbatch_tensor[:, :-1]
        y_gt = mbatch_tensor[:, 1:]

        with ctx:
            y_hat, (mbatch_loss, mbatch_acc, mbatch_acc_hit_ids) = model(x, y_gt)

            mbatch_loss = mbatch_loss / len(batch)
            mbatch_acc = mbatch_acc / len(batch)

        if not in_eval:
            scaler.scale(mbatch_loss).backward()

        batch_loss += mbatch_loss
        batch_acc += mbatch_acc

        batch_acc_hit_ids = batch_acc_hit_ids.union(set(mbatch_acc_hit_ids.tolist()))

    if in_eval:
        model.train()

    return batch_loss, batch_acc, batch_acc_hit_ids

def t_v_status(t0, iter_idx, n_iter, batch_loss, batch_acc, batch_unique_ids, batch_proc_duration, in_eval=False):
    mode = 'V' if in_eval else 'T'

    elapsed = time.time() - t0

    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))

    print(f"{elapsed_str} | {mode} Step {iter_idx:7} of {n_iter} | {mode} loss {batch_loss:2.5} | {mode} acc {batch_acc:2.5} | {mode} unique tokens: {len(batch_unique_ids):6} | {mode} batch time: {batch_proc_duration:2.5}")

# copied from nanoGPT
def get_lr(it, min_lr, max_lr, warmup_iters, lr_decay_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)
    

def main():
    assert torch.cuda.is_available(), "CUDA is not available"

    t0 = time.time()

    cfg = CFG

    torch.manual_seed(cfg.rng_seed)

    if cfg.dtype_name == "float32":
        torch.set_float32_matmul_precision('high')

    ddp_device = None

    if cfg.ddp:
        init_process_group(backend=cfg.ddp_backend)
        cfg.device = f"{cfg.device}:{cfg.ddp_local_rank}"

        torch.cuda.set_device(cfg.device)

    tok = load_tokenizer()

    tcfg = TrafoConfig(embed_dim=768, ctx_size=1024, vocab_size=tok.get_vocab_size(), num_blocks=12, num_heads=12, qkv_bias=True, ff_bias=True, is_causal=True)

    model_raw = Trafo(tcfg).to(cfg.device)
    model_compiled = torch.compile(model_raw)

    autocast_ctx = nullcontext() if cfg.device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=cfg.dtype)

    model = None
    if cfg.ddp:
        model = DDP(model_compiled)
    else:
        model = model_compiled

    ds_iter = load_dataset(cfg.ds_quick, cfg.rng_seed)

    raw_batch_iter = dataset_batch_iter(ds_iter, tok, cfg.batch_size, cfg.mbatch_size, tcfg.ctx_size)

    batch_iter = NBBatchQueue(raw_batch_iter, cfg.max_batch_prefetch)

    del raw_batch_iter


    all_params = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    decay_params = [p for n, p in all_params.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in all_params.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.optim_weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]

    opt = torch.optim.AdamW(optim_groups, lr=cfg.max_lr, betas=(0.9, 0.95), fused=True)

    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.dtype_name != "float32"))

    v_batch = batch_iter.next_item()

    if cfg.chart and cfg.master_process:
        ani = animation.FuncAnimation(CHART_DATA.fig, CHART_DATA.plot, interval=1000, blit=False)

        plt.show(block=False)

    for iter_idx in range(cfg.n_iter):
        if cfg.chart and cfg.master_process:
            plt.pause(0.25)


        lr = get_lr(iter_idx, cfg.min_lr, cfg.max_lr, cfg.warmup_iters, cfg.n_iter)

        for param_group in opt.param_groups:
            param_group['lr'] = lr


        batch_load_start = time.perf_counter()
        t_batch = batch_iter.next_item()
        batch_load_time = time.perf_counter() - batch_load_start

        if t_batch is None:
            print("Batch queue exhausted, bailing out...")
            break

        if cfg.debug_perf:
            rank = cfg.ddp_rank if cfg.ddp else 0
            print(f"Rank {rank}: batch load took {batch_load_time:3.5}s")
            
        t_start = time.perf_counter()

        batch_loss, batch_acc, batch_acc_hit_ids = t_v_step(model, scaler, autocast_ctx, cfg.device, t_batch, cfg.ddp, cfg.master_process, False)

        if cfg.master_process:
            elapsed = time.perf_counter() - t_start
            t_v_status(t0, iter_idx, cfg.n_iter, batch_loss.item(), batch_acc.item(), sorted(list(batch_acc_hit_ids)), elapsed, False)

        CHART_DATA.update(batch_loss.tolist(), batch_acc.tolist(), [len(batch_acc_hit_ids)])

        del batch_loss, batch_acc, batch_acc_hit_ids

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(opt)
        scaler.update()

        opt.zero_grad(set_to_none=True)

        if (iter_idx % cfg.v_interval == 0) and cfg.master_process:
            v_start = time.perf_counter()

            v_start = time.perf_counter()
            v_batch_loss, v_batch_acc, v_batch_acc_hit_ids = t_v_step(model, scaler, autocast_ctx, cfg.device, v_batch, cfg.master_process, True)

            elapsed = time.perf_counter() - v_start
            t_v_status(t0, iter_idx, cfg.n_iter, v_batch_loss.item(), v_batch_acc.item(), sorted(list(v_batch_acc_hit_ids)), elapsed, True)

            CHART_DATA.update([], [], [], v_batch_loss.tolist(), v_batch_acc.tolist(), [len(v_batch_acc_hit_ids)])

        if (iter_idx % cfg.gen_interval == 0) and cfg.master_process:
            tokens = model_compiled.generate(tok.encode("My answer to your question about apples is:").ids, cfg.gen_n_tokens, tok.encode("<|endoftext|>").ids[0], cfg.gen_topk_k, cfg.gen_temperature, cfg.device)

            decoded = tok.decode(tokens)

            print(f"Gen: {decoded}")

        if cfg.quitting:
            break

    if cfg.chart and cfg.master_process:
        CHART_DATA.save_plot()

    if cfg.ddp:
        destroy_process_group()

    print("main(): Bye!")
    sys.exit(0)

if __name__ == "__main__":
    main()
