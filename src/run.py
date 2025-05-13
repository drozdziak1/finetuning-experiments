from tinygrad import Context, Device, dtypes, nn
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import trange
from tinygrad.nn import LayerNorm, Linear, Embedding
from tinygrad.nn.optim import AdamW, Optimizer, OptimizerGroup
from tinygrad.tensor import Tensor

from os import getenv
from tokenizers import Tokenizer
from typing import List

from lr_scheduler import CosineAnnealingLR, LR_Scheduler

import datasets
import ipdb
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import polars
import random
import signal
import sys
import threading
import time


N_BATCHES = 600_000

V_INTERVAL = 50
GEN_INTERVAL = 50

GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(int(getenv("GPUS", 2))))

# Whether to use just the first parquet file or the whole dataset (the whole dataset takes ~30s to load)
DS_QUICK = bool(getenv("DS_QUICK", False))

# Whether to draw the charts window
CHART = bool(getenv("CHART", False))

BATCH_SIZE = 128

MINIBATCH_SIZE = 4

CTX_SIZE = 1024
NUM_BLOCKS = 12
EMBED_DIM = 768
NUM_HEADS = 12
FF_DIM = 4 * EMBED_DIM
DROPOUT = 0

LR = 6e-4
WARMUP_LR = 6e-5
WARMUP_ROUNDS = 2000

EPSILON = 1e-7

MAX_NORM = 1.0 # Max l2 norm for gradients

TOPK_K = 1

N_GEN_TOKENS = 8

RNG_SEED = 0xdeadbeef
random.seed(RNG_SEED)

Tensor.manual_seed(RNG_SEED)

NP_RNG = np.random.default_rng(seed=RNG_SEED)

# determined at runtime from vocab_size + 1
ACC_MISS_ID = None


class LRSchedWithWarmup:
    """
    Shoves a bunch of warmup steps before the specified Scheduler begins its first step
    """

    def __init__(self, warmup_rounds: int, start_lr: float, next_lr: float, opt: Optimizer, next_scheduler_ctor, *args, **kwargs):
        self.warmup_rounds = warmup_rounds
        self.next_lr = next_lr
        self.opt = opt
        self.cur_step = 1

        self.start_lr = start_lr

        self._set_lr(start_lr)

        self.next_scheduler_ctor = next_scheduler_ctor
        self.next_scheduler_args = args
        self.next_scheduler_kwargs = kwargs

    def _set_lr(self, new_lr):
        opts = []

        if isinstance(self.opt, OptimizerGroup):
            opts = self.opt.optimizers
        else:
            opts = [self.opt]
        
        for opt in opts:
            opt.lr.assign([new_lr])


    def step(self, *args, **kwargs):
        if self.cur_step <= self.warmup_rounds:
            new_lr = self.next_lr * self.cur_step / self.warmup_rounds

            self._set_lr(new_lr)

        if self.cur_step == self.warmup_rounds:
            self.opt.lr.assign(Tensor([self.next_lr], requires_grad=False, device=self.opt.device, dtype=self.opt.lr.dtype))
            self.next_scheduler = self.next_scheduler_ctor(*self.next_scheduler_args, **self.next_scheduler_kwargs)

        if self.cur_step >= self.warmup_rounds:
            self.next_scheduler.step(*args, **kwargs)

        self.cur_step += 1


class ChartData:
    def __init__(self):
        self.fig, (self.ax_loss, self.ax_acc, self.ax_acc_unique_hits, self.ax_layer_preview) = plt.subplots(4, 1, height_ratios=[1, 1, 1, 3])

        self.t_loss = np.array([])
        self.t_acc = np.array([])
        self.t_acc_unique_hits = np.array([])
        self.v_loss = np.array([])
        self.v_acc = np.array([])
        self.v_acc_unique_hits = np.array([])

        self.ax_loss.set_title("Loss")
        self.ax_loss.set_xlabel("training batch")
        self.ax_loss.set_ylabel("cat x-entropy loss")

        self.ax_acc.set_title("Accuracy")
        self.ax_acc.set_xlabel("training batch")
        self.ax_acc.set_ylabel("accuracy")

        self.ax_acc_unique_hits.set_title("Unique accuracy hits")
        self.ax_acc_unique_hits.set_xlabel("training batch")
        self.ax_acc_unique_hits.set_ylabel("n unique correct tokens")

        self.ax_layer_preview.set_title("Excerpt from embed layer")

        self.ln_t_loss, self.ln_v_loss = self.ax_loss.plot([], self.t_loss, 'b-', [], [], 'r--')
        self.ln_t_acc, self.ln_v_acc = self.ax_acc.plot([], [], 'b-', [], [], 'r--')
        self.ln_t_acc_unique_hits, self.ln_v_acc_unique_hits = self.ax_acc_unique_hits.plot([], [], 'b-', [], [], 'r--')


    def plot(self, _i):
        t_idx = np.arange(len(self.t_loss))
        v_idx = np.arange(len(self.v_loss)) * V_INTERVAL

        self.ln_t_loss.set_data(t_idx, self.t_loss)
        self.ln_v_loss.set_data(v_idx, self.v_loss)

        self.ln_t_acc.set_data(t_idx, self.t_acc)
        self.ln_v_acc.set_data(v_idx, self.v_acc)

        self.ln_t_acc_unique_hits.set_data(t_idx, self.t_acc_unique_hits)
        self.ln_v_acc_unique_hits.set_data(v_idx, self.v_acc_unique_hits)

        self.ax_loss.relim()
        self.ax_loss.autoscale()
        self.ax_acc.relim()
        self.ax_acc.autoscale()
        self.ax_acc_unique_hits.relim()
        self.ax_acc_unique_hits.autoscale()


CHART_DATA = ChartData()

def save_plot():
    try:
        plot_fname = "last_plot.png"
        print(f"Saving plot to {plot_fname}...")

        CHART_DATA.fig.savefig(plot_fname)
        plt.close(CHART_DATA.fig)

    except Exception as e:
        print(f"Could not save plot:\n{e}")

QUITTING = False

def sig_handler(_i, _whatever):
    global QUITTING
    if QUITTING:
        print("Quitting now!")
        sys.exit(1)
    print(f"Cleaning up... (Send SIGINT or SIGTERM again to quit now)")
    QUITTING = True

# signal.signal(signal.SIGINT, sig_handler)
# signal.signal(signal.SIGTERM, sig_handler)

class Transformer:
    def __init__(self, ctx_size, num_blocks, embed_dim, num_heads, ff_dim, vocab_size, dropout):
        assert embed_dim % num_heads == 0

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.ctx_size = ctx_size
        self.vocab_size = vocab_size + (64 - vocab_size % 64) # Round up to nearest multiple of 64
        self.dropout = dropout

        self.embed = Embedding(self.vocab_size, embed_dim)

        self.pe = Embedding(ctx_size, embed_dim)


        self.blocks = [TBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)]

        self.ln = LayerNorm(embed_dim, eps=EPSILON)

        self.unembed = Linear(embed_dim, self.vocab_size, bias=False)

        self.embed.weight = self.unembed.weight

        for pn, p in nn.state.get_state_dict(self).items():
            if pn.endswith("atn_out_w") or pn.endswith("ff2.weight"):
                print(f"Setting {pn} using c_proj rule")
                p.assign(Tensor.normal(p.shape, mean=0.0, std=0.02/math.sqrt(2 * self.num_blocks), dtype=p.dtype, device=p.device))
            # elif pn.endswith("bias"):
            #     print(f"Setting {pn} using bias rule")
            #     p.assign(p.zeros_like())
            # else:
            #     print(f"Setting {pn} using default rule")
            #     p.assign(Tensor.normal(p.shape, mean=0.0, std=0.02, dtype=p.dtype, device=p.device))
                

    def __call__(self, x: Tensor, y_gt: Tensor = None):
        """
        X: (B, T)
        """

        x_embedded = self.embed(x)

        batch_size, seq_len = x.shape

        allpos = Tensor.arange(0, CTX_SIZE).unsqueeze(0).to_(GPUS)
        allpos.requires_grad = False
        x_pe = x_embedded + self.pe(allpos)

        x_refined = x_pe.sequential(self.blocks)

        x_ln = self.ln(x_refined)

        x_unembed = self.unembed(x_ln)

        if y_gt is not None:
            loss = x_unembed.sparse_categorical_crossentropy(y_gt)
            return x_unembed, loss

        return x_unembed

    def optim_init(self, lr, beta1, beta2, wd):
        param_dict = nn.state.get_state_dict(self)

        wd_params = []
        nowd_params = []

        for pn, p in param_dict.items():
            if len(p.shape) >= 2:
                wd_params.append(p)
            else:
                nowd_params.append(p)
                
        wd_opt = AdamW(wd_params, lr, b1=beta1, b2=beta2, weight_decay=wd)
        nowd_opt = AdamW(nowd_params, lr, b1=beta1, b2=beta2, weight_decay=0.0)

        return OptimizerGroup(wd_opt, nowd_opt)

class TBlock:
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.ln_attn = LayerNorm(embed_dim, eps=EPSILON)

        self.w_qkv = Linear(embed_dim, 3 * embed_dim, bias=False)

        self.atn_out_w = Tensor.kaiming_normal(embed_dim, embed_dim)

        self.ln_ff = LayerNorm(embed_dim, eps=EPSILON)

        self.ff1 = Linear(embed_dim, ff_dim)
        self.ff2 = Linear(ff_dim, embed_dim)


    def __call__(self, x: Tensor):
        """
        X: (B, T, C)
        """

        x_ln_attn = self.ln_attn(x)

        x_qkv = self.w_qkv(x_ln_attn)

        x_q, x_k, x_v = [x_qkv.shrink((None, None, (i*self.embed_dim, (i+1) * self.embed_dim))).reshape(None, None, self.num_heads, self.head_size).transpose(1, 2) for i in range(3)]

        x_atn = Tensor.scaled_dot_product_attention(x_q, x_k, x_v, is_causal=True, dropout_p=DROPOUT)

        x_atn_out = x_atn.transpose(1, 2).reshape(x.shape).linear(self.atn_out_w)

        x = x + x_atn_out.dropout(self.dropout)

        x_ln_ff = self.ln_ff(x)

        x_ff1 = self.ff1(x_ln_ff).gelu()

        x_ff2 = self.ff2(x_ff1)

        x = x + x_ff2.dropout(self.dropout)

        return x


def load_tokenizer(model_name="gpt2"):
    return Tokenizer.from_pretrained(model_name)

def load_dataset():
    if DS_QUICK:
        ds = datasets.load_dataset("HuggingFaceFW/fineweb", data_files="sample/100BT/000_00000.parquet", split="train", )
    else:
        ds = datasets.load_dataset("HuggingFaceFW/fineweb", "sample-100BT", split="train")

    ds_iter = ds.to_iterable_dataset().shuffle(buffer_size=10_000, seed=RNG_SEED).iter(1)

    return ds_iter

def dataset_batch_iter(ds_iter, tokenizer, batch_size, minibatch_size, ctx_size=CTX_SIZE):
    """
    each iteration yields a training batch
    """
    bufs = [[[] for _ in range(minibatch_size)] for _ in range(batch_size)]
    while True:
        batch = []
        for b_i in range(batch_size):
            minibatch = []
            for mb_i in range(minibatch_size):
                while len(bufs[b_i][mb_i]) < ctx_size + 1:
                    bufs[b_i][mb_i] += tokenizer.encode("<|endoftext|>").ids
                    txt = next(ds_iter)["text"][0]
                    bufs[b_i][mb_i] += tokenizer.encode(txt).ids

                minibatch_item = bufs[b_i][mb_i][:ctx_size + 1]

                bufs[b_i][mb_i] = bufs[b_i][mb_i][ctx_size + 1:]

                minibatch += [minibatch_item]
            batch += [minibatch]

        yield batch

def clip_grad_norm(opt, max_norm=1.0):
    l2_sum = Tensor(0.0, device=opt.params[0].device, dtype=opt.params[0].dtype, requires_grad=False)

    for p in opt.params:
        if p.grad is not None:
            l2_sum = l2_sum + p.grad.square().sum()


    l2_norm = l2_sum.sqrt()

    if l2_norm.item() > max_norm:
        factor = max_norm / l2_norm

        print(f"Clipping grads by factor of {factor.item():,}")

        for p in opt.params:
            if p.grad is not None:
                new_grad = p * factor
                new_grad.requires_grad = False
                p.grad.assign(new_grad)
    

def t_v_step(model: Transformer, batch: Tensor, opt: Optimizer, train: bool):
    @TinyJit
    def mb_step(model, x, y_gt):

        y_hat, loss = model(x, y_gt)

        loss = loss / BATCH_SIZE # Internally, x-entropy implementation is not too fp16-friendly, so we do the mean on our own

        loss.backward()

        acc_scores = y_hat.argmax(-1) == y_gt
        acc_scores.requires_grad = False

        acc_hits_by_id = acc_scores.where(y_gt, ACC_MISS_ID).reshape(-1).one_hot(model.vocab_size + 2).sum(0)
        acc_hits_by_id.requires_grad = False

        return loss, acc_scores, acc_hits_by_id

    if train:
        Tensor.training = True

    total_loss = Tensor(0.0, requires_grad=False).to_(GPUS)
    total_acc_hit_cnt = Tensor(0, requires_grad=False, dtype=dtypes.int32).to_(GPUS)

    total_acc_hits_by_id = Tensor.zeros(model.vocab_size + 2, requires_grad=False, dtype=dtypes.int32).shard_(GPUS)

    acc_cnt = BATCH_SIZE * MINIBATCH_SIZE * CTX_SIZE

    if train:
        opt.zero_grad()

    mb_iter_start = time.perf_counter()
    for i in trange(BATCH_SIZE):
        minibatch = batch[i]

        x = minibatch[:, :-1].shard_(GPUS).contiguous()
        y_gt = minibatch[:, 1:].shard_(GPUS).contiguous()

        loss, acc_scores, acc_hits_by_id = mb_step(model, x, y_gt)

        total_loss = total_loss + loss.realize()
        total_acc_hit_cnt = total_acc_hit_cnt + acc_scores.sum().realize()
        total_acc_hits_by_id = total_acc_hits_by_id + acc_hits_by_id.realize()

    mb_iter_end = time.perf_counter()
    print(f"fwd took {mb_iter_end - mb_iter_start:.5}s")

    if train:
        clip_grad_norm(opt, MAX_NORM)
        opt_step_start = time.perf_counter()
        opt.step()
        opt.zero_grad()
        opt_step_end = time.perf_counter()
        print(f"bwd took {opt_step_end - opt_step_start:.5}s")

    Tensor.training = False

    arange_helper = Tensor.arange(model.vocab_size + 2).shard_(GPUS)
    total_acc_hit_id_set = set((total_acc_hits_by_id > 0).where(arange_helper, ACC_MISS_ID).tolist())

    total_acc_hit_id_set.remove(ACC_MISS_ID)

    return total_loss.item(), total_acc_hit_cnt.item(), acc_cnt, total_acc_hit_id_set, total_acc_hits_by_id

def gen_step(model: Transformer, tokenizer: Tokenizer):
    @TinyJit
    def gen_token_step(model, tokens: Tensor) -> Tensor:
        ys = model(tokens)

        last_y = ys[0, len(tokens) - 1]

        _topk_values, topk_indices = last_y.topk(TOPK_K, dim=-1)

        return topk_indices

    x = tokenizer.encode("<|endoftext|>").ids * CTX_SIZE

    tokens = tokenizer.encode("My name is").ids

    for i in range(N_GEN_TOKENS):

        for j in range(len(tokens)):
            x[j] = tokens[j]

        topk_indices = gen_token_step(model, Tensor(x, requires_grad=False).unsqueeze(0).to_(GPUS))

        tok_id = NP_RNG.choice(topk_indices.reshape(-1).numpy())

        tokens += [tok_id]

    return tokenizer.decode(tokens)

def main():
    tok = load_tokenizer()


    ds_it = load_dataset()

    batch_it = dataset_batch_iter(ds_it, tok, BATCH_SIZE, MINIBATCH_SIZE)

    v_batch = Tensor(next(batch_it), requires_grad=False)

    model = Transformer(CTX_SIZE, NUM_BLOCKS, EMBED_DIM, NUM_HEADS, FF_DIM, tok.get_vocab_size(), DROPOUT)

    global ACC_MISS_ID
    ACC_MISS_ID = model.vocab_size + 1

    # Sharding
    for _k, x in nn.state.get_state_dict(model).items(): x.to_(GPUS)


    # opt = model.optim_init(LR, 0.9, 0.95, 0.1)
    opt = AdamW(nn.state.get_parameters(model), b1=0.9, b2=0.95, weight_decay=0.1)

    lr_sched = LRSchedWithWarmup(WARMUP_ROUNDS, WARMUP_LR, LR, opt, CosineAnnealingLR, opt, 100000)

    # I'm a bit tired of setting DEBUG=2 manually when playing with the  metaparameters
    first_pass = True

    ani = animation.FuncAnimation(CHART_DATA.fig, CHART_DATA.plot, interval=1000, blit=False)

    if CHART:
        plt.show(block=False)

    for i in range(N_BATCHES):
        if CHART:
            plt.pause(1)

        if QUITTING:
            break

        ctx = None
        if first_pass:
            ctx = Context(DEBUG=2)
        else:
            ctx = Context()

        with ctx:

            try:
                iter_start = time.perf_counter()
                batch_list = next(batch_it)
                iter_end = time.perf_counter()
                print(f"batch load took {iter_end - iter_start:.5}s")

                batch = Tensor(batch_list)

            except Exception as e:
                print(f"Ran out of batches for training! Cleaning up...")
                save_plot()
                sys.exit(1)

            t_start = time.perf_counter()
            t_loss, t_acc_hit_cnt, t_acc_cnt, t_acc_hit_id_set, t_acc_hits_by_id = t_v_step(model, batch, opt, True)
            t_end = time.perf_counter()

            print(f"train_step() took {t_end - t_start:.5}s")

            t_acc_hit_cnt = t_acc_hit_cnt

            t_acc = t_acc_hit_cnt / t_acc_cnt

            t_acc_hit_ids = sorted(list(t_acc_hit_id_set))

            t_hits_decoded = tok.decode(t_acc_hit_ids)

            CHART_DATA.t_loss = np.append(CHART_DATA.t_loss, [t_loss])
            CHART_DATA.t_acc = np.append(CHART_DATA.t_acc, [t_acc])
            CHART_DATA.t_acc_unique_hits = np.append(CHART_DATA.t_acc_unique_hits, [len(t_acc_hit_ids)])

            maybe_warmup = f"WARMUP {i+1}/{WARMUP_ROUNDS} | " if i < WARMUP_ROUNDS else ""

            print(f"{maybe_warmup}Step {i+1:10} of {N_BATCHES} | T Loss: {t_loss:20.10} | T acc: {t_acc_hit_cnt:6} of {t_acc_cnt:6} ({t_acc:20.10}) | T unique acc hits: {len(t_acc_hit_ids):4} {t_acc_hit_ids} {t_hits_decoded}")

            lr_sched.step()

            if i % V_INTERVAL == 0:

                v_start = time.perf_counter()
                v_loss, v_acc_hit_cnt, v_acc_cnt, v_acc_hit_id_set, _v_acc_hits_by_id = t_v_step(model, v_batch, opt, False)
                v_end = time.perf_counter()

                print(f"train_step() took {v_end - v_start:.5}s")

                v_acc_hit_cnt = v_acc_hit_cnt

                v_acc = v_acc_hit_cnt / v_acc_cnt

                v_acc_hit_ids = sorted(list(v_acc_hit_id_set))

                v_hits_decoded = tok.decode(v_acc_hit_ids)

                CHART_DATA.v_loss = np.append(CHART_DATA.v_loss, [v_loss])
                CHART_DATA.v_acc = np.append(CHART_DATA.v_acc, [v_acc])
                CHART_DATA.v_acc_unique_hits = np.append(CHART_DATA.v_acc_unique_hits, [len(v_acc_hit_ids)])

                maybe_warmup = f"WARMUP {i+1}/{WARMUP_ROUNDS} | " if i < WARMUP_ROUNDS else ""

                print(f"{maybe_warmup}Step {i+1:10} of {N_BATCHES} | V Loss: {v_loss:20.10} | V acc: {v_acc_hit_cnt:6} of {v_acc_cnt:6} ({v_acc:20.10}) | V unique acc hits: {len(v_acc_hit_ids):4} {v_acc_hit_ids} {v_hits_decoded}")

            if i % GEN_INTERVAL == 0:
                gen_str = gen_step(model, tok)

                print(f"Gen: {gen_str}")

            sys.stdout.flush()

        first_pass = False

    save_plot()

if __name__ == "__main__":
    main()
