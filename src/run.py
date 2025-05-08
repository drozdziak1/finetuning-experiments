from tinygrad import Context, Device, dtypes, nn
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import trange
from tinygrad.nn import LayerNorm, Linear, Embedding
from tinygrad.nn.optim import AdamW, Optimizer
from tinygrad.tensor import Tensor

from os import getenv
from tokenizers import Tokenizer
from typing import List

from lr_scheduler import CosineAnnealingLR, LR_Scheduler

import datasets
import ipdb
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

BATCH_SIZE = 64

MINIBATCH_SIZE = 8

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

TOPK_K = 16

N_GEN_TOKENS = 8

RNG_SEED = 0xdeadbeef
random.seed(RNG_SEED)

Tensor.manual_seed(RNG_SEED)

NP_RNG = np.random.default_rng(seed=RNG_SEED)


class LRSchedWithWarmup:
    """
    Shoves a bunch of warmup steps before the specified Scheduler begins its first step
    """

    def __init__(self, warmup_rounds: int, next_lr: float, opt: Optimizer, next_scheduler_ctor, *args, **kwargs):
        self.warmup_rounds = warmup_rounds
        self.next_lr = next_lr
        self.opt = opt
        self.cur_step = 0

        self.start_lr = self.opt.lr

        self.next_scheduler_ctor = next_scheduler_ctor
        self.next_scheduler_args = args
        self.next_scheduler_kwargs = kwargs


    def step(self, *args, **kwargs):
        if self.cur_step <= self.warmup_rounds:
            new_lr = (self.next_lr - self.start_lr) * self.cur_step / self.warmup_rounds

            self.opt.lr.assign(new_lr)

        if self.cur_step == self.warmup_rounds:
            self.opt.lr.assign(Tensor([self.next_lr], requires_grad=False, device=self.opt.device, dtype=self.opt.lr.dtype))
            self.next_scheduler = self.next_scheduler_ctor(*self.next_scheduler_args, **self.next_scheduler_kwargs)

        if self.cur_step >= self.warmup_rounds:
            self.next_scheduler.step(*args, **kwargs)

        self.cur_step += 1


class ChartData:
    def __init__(self):
        self.fig, (self.ax_loss, self.ax_acc, self.ax_acc_unique_hits) = plt.subplots(3)

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

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

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

        self.unembed = Tensor.kaiming_normal(embed_dim, self.vocab_size)

    def __call__(self, x: Tensor):
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

        x_unembed = x_ln.dot(self.unembed).softmax()

        return x_unembed






class TBlock:
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.ln_attn = LayerNorm(embed_dim, eps=EPSILON)

        self.w_qkv = Linear(embed_dim, 3 * embed_dim, bias=True)

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

        # x_q, x_k, x_v = [x_ln_attn.matmul(t).reshape(x_ln_attn.shape[0], x_ln_attn.shape[1], self.num_heads, self.head_size).transpose(1, 2) for t in [self.q_w, self.k_w, self.v_w]]

        x_atn = Tensor.scaled_dot_product_attention(x_q, x_k, x_v, is_causal=True)

        x_atn_out = x_atn.transpose(1, 2).reshape(x.shape).matmul(self.atn_out_w)

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

def dataset_minibatch_iter(ds_iter, tokenizer, minibatch_size, ctx_size=CTX_SIZE):
    """
    each iteration yields a training batch
    """
    bufs = [tokenizer.encode(next(ds_iter)["text"][0]).ids for _ in range(minibatch_size)]
    while True:
        batch = []

        for buf in bufs:
            while len(buf) < ctx_size + 1:
                buf += tokenizer.encode("<|endoftext|>").ids
                buf += tokenizer.encode(next(ds_iter)["text"][0]).ids

            batch_item = buf[:ctx_size + 1]
            buf = buf[ctx_size + 1:]

            batch += [batch_item]

        yield batch

def t_v_step(model: Transformer, batch: Tensor, opt: Optimizer, train: bool):
    @TinyJit
    def mb_step(model, x, y_gt):
        y_hat = model(x)

        losses = y_hat.sparse_categorical_crossentropy(y_gt, reduction = "none")

        loss = losses.mean() / BATCH_SIZE # Internally, x-entropy implementation is not too fp16-friendly, so we do the mean on our own

        loss.backward()

        acc_scores = y_hat.argmax(-1) == y_gt
        acc_scores.requires_grad = False

        acc_hits_by_id = acc_scores.where(y_gt, model.vocab_size + 1).reshape(-1)
        acc_hits_by_id.requires_grad = False

        return loss, acc_scores.sum(dtype=dtypes.int32), acc_hits_by_id

    if train:
        Tensor.training = True

    total_loss = Tensor(0.0, requires_grad=False).to_(GPUS)
    total_acc_hit_cnt = Tensor(0, requires_grad=False, dtype=dtypes.int32).to_(GPUS)

    total_acc_hits_by_id = set()

    # For assigning token ID value during set construction
    arange_helper = Tensor.arange(0, model.vocab_size + 2, requires_grad=False).shard_(GPUS, axis=0)

    acc_cnt = BATCH_SIZE * MINIBATCH_SIZE * CTX_SIZE

    if train:
        opt.zero_grad()

    mb_iter_start = time.perf_counter()
    for i in trange(BATCH_SIZE):

        minibatch = batch[i]

        x = minibatch[:, :-1].shard_(GPUS, axis=0)
        y_gt = minibatch[:, 1:].shard_(GPUS, axis=0)

        loss, acc_hit_cnt, acc_hits_by_id = mb_step(model, x, y_gt)

        total_loss = total_loss + loss
        total_acc_hit_cnt = total_acc_hit_cnt + acc_hit_cnt
        total_acc_hits_by_id = total_acc_hits_by_id.union(acc_hits_by_id.tolist())

    mb_iter_end = time.perf_counter()
    print(f"fwd took {mb_iter_end - mb_iter_start:.5}s")

    if train:
        opt_step_start = time.perf_counter()
        opt.step()
        opt_step_end = time.perf_counter()
        print(f"bwd took {opt_step_end - opt_step_start:.5}s")

    Tensor.training = False

    total_acc_hits_by_id.remove(model.vocab_size + 1)

    return total_loss.item(), total_acc_hit_cnt.item(), acc_cnt, total_acc_hits_by_id

def gen_step(model: Transformer, tokenizer: Tokenizer):
    @TinyJit
    def gen_token_step(model, tokens: Tensor) -> Tensor:
        ys = model(tokens)

        last_y = ys[0, len(tokens) - 1]

        _topk_values, topk_indices = last_y.topk(TOPK_K, dim=-1)

        return topk_indices

    x = tokenizer.encode("<|endoftext|>").ids * CTX_SIZE

    tokens = tokenizer.encode("What in").ids

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

    minibatch_it = dataset_minibatch_iter(ds_it, tok, MINIBATCH_SIZE)

    v_batch = Tensor([next(minibatch_it) for i in range(BATCH_SIZE)], requires_grad=False)

    model = Transformer(CTX_SIZE, NUM_BLOCKS, EMBED_DIM, NUM_HEADS, FF_DIM, tok.get_vocab_size(), DROPOUT)

    # Sharding
    for _k, x in nn.state.get_state_dict(model).items(): x.to_(GPUS)

    params = nn.state.get_parameters(model)

    param_count = sum(map(lambda t: len(t.reshape(-1)), params))

    print(f"Training a {param_count} parameter model")

    opt = AdamW(params=params, lr=WARMUP_LR, b2=0.95, weight_decay=1e-1)
    lr_sched = LRSchedWithWarmup(WARMUP_ROUNDS, LR, opt, CosineAnnealingLR, opt, 100000)

    # I'm a bit tired of setting DEBUG=2 manually when playing with the  metaparameters
    first_pass = True

    ani = animation.FuncAnimation(CHART_DATA.fig, CHART_DATA.plot, interval=1000, blit=False)

    if CHART:
        plt.show(block=False)

    for i in range(N_BATCHES):
        if CHART:
            plt.pause(0.05)

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
                batch_list = [next(minibatch_it) for _ in range(BATCH_SIZE)]
                iter_end = time.perf_counter()
                print(f"batch load took {iter_end - iter_start:.5}s")

                batch = Tensor(batch_list)

            except Exception as e:
                print(f"Ran out of batches for training! Cleaning up...")
                save_plot()
                sys.exit(1)

            t_start = time.perf_counter()
            t_loss, t_acc_hit_cnt, t_acc_cnt, t_acc_hit_ids = t_v_step(model, batch, opt, True)
            t_end = time.perf_counter()

            print(f"train_step() took {t_end - t_start:.5}s")

            t_acc_hit_cnt = t_acc_hit_cnt

            t_acc = t_acc_hit_cnt / t_acc_cnt

            t_acc_hit_ids = sorted(list(t_acc_hit_ids))

            t_hits_decoded = tok.decode(t_acc_hit_ids)

            CHART_DATA.t_loss = np.append(CHART_DATA.t_loss, [t_loss])
            CHART_DATA.t_acc = np.append(CHART_DATA.t_acc, [t_acc])
            CHART_DATA.t_acc_unique_hits = np.append(CHART_DATA.t_acc_unique_hits, [len(t_acc_hit_ids)])

            maybe_warmup = f"WARMUP {i+1}/{WARMUP_ROUNDS} | " if i < WARMUP_ROUNDS else ""

            print(f"{maybe_warmup}Step {i+1:10} of {N_BATCHES} | T Loss: {t_loss:20.10} | T acc: {t_acc_hit_cnt:6} of {t_acc_cnt:6} ({t_acc:20.10}) | T unique acc hits: {len(t_acc_hit_ids):4} {t_acc_hit_ids} {t_hits_decoded}")

            lr_sched.step()

            if i % V_INTERVAL == 0:

                v_start = time.perf_counter()
                v_loss, v_acc_hit_cnt, v_acc_cnt, v_acc_hit_ids = t_v_step(model, v_batch, opt, False)
                v_end = time.perf_counter()

                print(f"train_step() took {v_end - v_start:.5}s")

                v_acc_hit_cnt = v_acc_hit_cnt

                v_acc = v_acc_hit_cnt / v_acc_cnt

                v_acc_hit_ids = sorted(list(v_acc_hit_ids))

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
