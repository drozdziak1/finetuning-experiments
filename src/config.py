import torch

from dataclasses import dataclass

import os

dtype_name2dtype = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}

@dataclass
class Config:
    min_lr: float = 6e-5
    max_lr: float = 6e-4
    warmup_iters: int = 2_000
    n_iter: int = 600_000
    optim_weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    ds_quick: bool = bool(os.getenv("DS_QUICK", False))
    chart: bool = bool(os.getenv("CHART", False))
    device: str = os.getenv("DEVICE", "cuda")
    ddp: bool = int(os.getenv("RANK", -1)) != -1
    ddp_rank: int = int(os.getenv("RANK", -1))
    ddp_local_rank = int(os.getenv("LOCAL_RANK", -1))
    ddp_world_size = int(os.getenv("WORLD_SIZE", -1))
    ddp_backend = os.getenv("DDP_BACKEND", "nccl")
    master_process: bool = ddp_rank == 0 or not ddp
    rng_seed: int = 0xdeadbeef + ddp_rank + 1
    v_interval: int = 50
    gen_interval: int = 50
    batch_size: int = 20 // (ddp_world_size if ddp else 1)
    mbatch_size: int = 24
    dtype_name: str = os.getenv("DTYPE", "bfloat16")
    dtype = dtype_name2dtype[dtype_name]
    gen_n_tokens: int = 128
    gen_topk_k: int = 50
    gen_temperature: float = 0.7
    quitting: bool = False
    max_batch_prefetch: int = 4
    debug_perf: bool = bool(os.getenv("DEBUG_PERF", False))


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
