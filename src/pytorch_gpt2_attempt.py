import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import sys

from torch import Tensor
from tqdm import tqdm

from config import TrafoConfig
from data_utils import load_dataset, load_tokenizer, dataset_batch_iter
from my_globals import CHART_DATA, CFG, QUITTING

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
        
class TrafoBlock(nn.Module):
    def __init__(self, cfg: TrafoConfig):
        super(TrafoBlock, self).__init__()
        self.cfg = cfg

        self.ln_atn = nn.LayerNorm(self.cfg.embed_dim)

        self.atn = MHSA(cfg)

        self.ln_ff = nn.LayerNorm(self.cfg.embed_dim)
        

        self.ff1 = nn.Linear(self.cfg.embed_dim, self.cfg.ff_dim)
        self.ff2 = nn.Linear(self.cfg.ff_dim, self.cfg.embed_dim)

        nn.init.normal_(self.ff2.weight, mean=0.0, std=0.02/math.sqrt(2 * self.cfg.num_blocks))

        
    def forward(self, xs):
        x_ln_atn = self.ln_atn(xs)
        x_atn = self.atn(x_ln_atn)

        x_ln_ff = self.ln_ff(xs)
        x_ff1 = F.gelu(self.ff1(x_ln_ff))
        x_ff2 = self.ff2(x_ff1)

        return xs + x_atn + x_ff2

        
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

def t_v_step(model, scaler, device, batch, in_eval=False):
    torch.set_grad_enabled(not in_eval)
    if in_eval:
        model.eval()

    batch_loss = torch.zeros((), device=device)
    batch_acc = torch.zeros((), device=device)

    batch_acc_hit_ids = set()

    for mbatch in tqdm(batch, unit="mbatches"):
        mbatch_tensor = torch.tensor(mbatch, dtype=torch.long, device=device)

        x = mbatch_tensor[:, :-1]
        y_gt = mbatch_tensor[:, 1:]

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

def t_v_status(iter_idx, n_iter, batch_loss, batch_acc, batch_unique_ids, tok, in_eval=False):
    mode = 'V' if in_eval else 'T'

    decoded = [tok.decode([t_id]) for t_id in batch_unique_ids]

    print(f"{mode} Step {iter_idx:7} of {n_iter} | {mode} loss {batch_loss:2.5} | {mode} acc {batch_acc:1.5} | {mode} unique tokens: {len(batch_unique_ids):6} {decoded}")

def get_lr(it, learning_rate, warmup_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
    

def main():
    assert torch.cuda.is_available(), "CUDA is not available"

    cfg = CFG

    torch.manual_seed(cfg.rng_seed)

    torch.set_float32_matmul_precision('high')

    tok = load_tokenizer()

    tcfg = TrafoConfig(embed_dim=768, ctx_size=1024, vocab_size=tok.get_vocab_size(), num_blocks=12, num_heads=12, qkv_bias=True, ff_bias=True, is_causal=True)

    model = torch.compile(Trafo(tcfg).to(cfg.device))
    # model = Trafo(tcfg).to(cfg.device)

    ds_iter = load_dataset(cfg.ds_quick, cfg.rng_seed)

    batch_iter = dataset_batch_iter(ds_iter, tok, cfg.batch_size, cfg.mbatch_size, tcfg.ctx_size)

    all_params = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    decay_params = [p for n, p in all_params.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in all_params.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.optim_weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]

    opt = torch.optim.AdamW(optim_groups, lr=cfg.lr, betas=(0.9, 0.95), fused=True)

    scaler = torch.amp.GradScaler('cuda',)

    v_batch = next(batch_iter)

    if cfg.chart:
        ani = animation.FuncAnimation(CHART_DATA.fig, CHART_DATA.plot, interval=1000, blit=False)

        plt.show(block=False)

    for iter_idx in range(cfg.n_iter):
        if cfg.chart:
            plt.pause(0.25)

        if QUITTING:
            CHART_DATA.save_plot()
            sys.exit(0)

        lr = get_lr(iter_idx, cfg.lr, cfg.warmup_iters, cfg.min_lr)

        for param_group in opt.param_groups:
            param_group['lr'] = lr

        batch = next(batch_iter)

        batch_loss, batch_acc, batch_acc_hit_ids = t_v_step(model, scaler, cfg.device, batch, False)

        t_v_status(iter_idx, cfg.n_iter, batch_loss.item(), batch_acc.item(), sorted(list(batch_acc_hit_ids)), tok, False)

        CHART_DATA.update(batch_loss.tolist(), batch_acc.tolist(), [len(batch_acc_hit_ids)])

        del batch_loss

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(opt)
        scaler.update()

        opt.zero_grad(set_to_none=True)

        if iter_idx % cfg.v_interval == 0:
            v_batch_loss, v_batch_acc, v_batch_acc_hit_ids = t_v_step(model, scaler, cfg.device, v_batch, True)

            t_v_status(iter_idx, cfg.n_iter, v_batch_loss.item(), v_batch_acc.item(), sorted(list(v_batch_acc_hit_ids)), tok, True)

            CHART_DATA.update([], [], [], v_batch_loss.tolist(), v_batch_acc.tolist(), [len(v_batch_acc_hit_ids)])


if __name__ == "__main__":
    main()
