import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch import Tensor

from dataclasses import dataclass

from run import load_dataset, load_tokenizer, dataset_batch_iter

@dataclass
class Config:
    batch_size: int = 40
    mbatch_size: int = 12
    lr: float = 6e-4
    optim_weight_decay: float = 0.1

@dataclass
class TrafoConfig:
    embed_dim: int
    ctx_size: int
    vocab_size: int
    num_blocks: int
    num_heads: int
    qkv_bias: bool
    ff_bias: bool
    is_causal: bool

    @property
    def head_dim(self):
        assert self.embed_dim % self.num_heads == 0, "num_heads must integer-divide embed_dim cleanly"
        return self.embed_dim // self.num_heads

    @property
    def ff_dim(self):
        return 4 * self.embed_dim
    

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

        loss = None

        if y_gt is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_gt.reshape(-1), ignore_index=-1)

        return logits, loss

# A lazy sanity check
def main():
    assert torch.cuda.is_available(), "CUDA is not available"
    device = "cuda"
    torch.set_float32_matmul_precision('high')

    cfg = Config()

    tok = load_tokenizer()

    tcfg = TrafoConfig(embed_dim=768, ctx_size=1024, vocab_size=tok.get_vocab_size(), num_blocks=12, num_heads=12, qkv_bias=True, ff_bias=True, is_causal=True)

    model = torch.compile(Trafo(tcfg).to(device))
    # model = Trafo(tcfg).to(device)

    ds_iter = load_dataset()

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

    for idx in range(600_000):
        batch = next(batch_iter)

        opt.zero_grad()

        batch_loss = torch.zeros((), device=device)

        for mbatch in batch:
            mbatch_tensor = torch.tensor(mbatch, dtype=torch.long, device=device)

            x = mbatch_tensor[:, :-1]
            y_gt = mbatch_tensor[:, 1:]

            y_hat, _mbatch_loss = model(x, y_gt)

            mbatch_loss = F.cross_entropy(y_hat.view(-1, y_hat.size(-1)), y_gt.reshape(-1), ignore_index=-1)

            mbatch_loss = mbatch_loss / cfg.batch_size

            scaler.scale(mbatch_loss).backward()

            batch_loss += mbatch_loss

        print(f"Step {idx} T loss: {batch_loss.item()}")

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(opt)
        scaler.update()



if __name__ == "__main__":
    main()
