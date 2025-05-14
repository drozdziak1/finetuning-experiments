from dataclasses import dataclass

import os

@dataclass
class Config:
    batch_size: int = 40
    mbatch_size: int = 12
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 2_000
    n_iter: int = 600_000
    optim_weight_decay: float = 0.1
    ds_quick: bool = bool(os.getenv("DS_QUICK", False))
    chart: bool = bool(os.getenv("CHART", False))
    device: str = os.getenv("DEVICE", "cuda")
    rng_seed: int = 0xdeadbeef
    v_interval: int = 50
    gen_inverval: int = 50

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
