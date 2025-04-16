from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW, Optimizer
from tinygrad.nn.state import get_parameters
from tinygrad.engine.jit import TinyJit
from tinygrad import Context, Device, dtypes
from tinygrad.nn import LayerNorm, Linear, Embedding

from tokenizers import Tokenizer
from typing import List

from lr_scheduler import CosineAnnealingLR

import ipdb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import polars
import random
import signal
import sys
import threading


FINEWEB_PATH = "/home/drozdziak1/Documents/datasets/fineweb/000_00000.parquet"

N_BATCHES = 16384 * 16384

N_V_BATCHES = 1
V_INTERVAL = 50

MINIBATCH_SIZE = 256
CTX_SIZE = 64

WARMUP_ROUNDS = MINIBATCH_SIZE * CTX_SIZE // 512

NUM_BLOCKS = 12
EMBED_DIM = 768
NUM_HEADS = 12
FF_DIM = 4 * EMBED_DIM // 3
DROPOUT = 0.1

LR = 1e-4

EPSILON = 1e-5

TOPK_K = 8

N_GEN_TOKENS = 8

RNG_SEED = 0xdeadbeef
random.seed(RNG_SEED)

Tensor.manual_seed(RNG_SEED)

NP_RNG = np.random.default_rng(seed=RNG_SEED)


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


class Transformer:
    def __init__(self, ctx_size, num_blocks, embed_dim, num_heads, ff_dim, vocab_size, dropout):
        assert embed_dim % num_heads == 0

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.ctx_size = ctx_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embed = Embedding(vocab_size, embed_dim)

        self.pe = Embedding(ctx_size, embed_dim)

        self.allpos = Tensor.arange(0, CTX_SIZE).unsqueeze(0)
        self.allpos.requires_grad = False

        self.blocks = [TBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)]

        self.ln = LayerNorm(embed_dim, eps=EPSILON)

        self.unembed = Tensor.kaiming_normal(embed_dim, vocab_size)

    def __call__(self, x: Tensor):
        """
        X: (B, T)
        """

        x_embedded = self.embed(x)

        batch_size, seq_len = x.shape

        x_pe = x_embedded + self.pe(self.allpos)

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

        self.q_w = Tensor.kaiming_normal(embed_dim, embed_dim)
        self.k_w = Tensor.kaiming_normal(embed_dim, embed_dim)
        self.v_w = Tensor.kaiming_normal(embed_dim, embed_dim)

        self.atn_out_w = Tensor.kaiming_normal(embed_dim, embed_dim)

        self.ln_ff = LayerNorm(embed_dim, eps=EPSILON)

        self.ff1 = Linear(embed_dim, ff_dim)
        self.ff2 = Linear(ff_dim, embed_dim)


    def __call__(self, x: Tensor):
        """
        X: (B, T, C)
        """

        x_ln_attn = self.ln_attn(x)

        x_q, x_k, x_v = [x_ln_attn.dot(t).reshape(x_ln_attn.shape[0], x_ln_attn.shape[1], self.num_heads, self.head_size) for t in [self.q_w, self.k_w, self.v_w]]

        x_atn = Tensor.scaled_dot_product_attention(x_q, x_k, x_v, is_causal=True)

        x_atn_out = x_atn.reshape(x.shape).dot(self.atn_out_w)

        x = x + x_atn_out.dropout(self.dropout)

        x_ln_ff = self.ln_ff(x)

        x_ff1 = self.ff1(x_ln_ff).gelu()

        x_ff2 = self.ff2(x_ff1)

        x = x + x_ff2.dropout(self.dropout)

        return x


def load_tokenizer(model_name="gpt2"):
    return Tokenizer.from_pretrained(model_name)

def load_dataset(data_path=FINEWEB_PATH):
    df = polars.read_parquet(data_path, columns=['text'])

    return df

def dataset_minibatch_iter(df, tokenizer, batch_size, ctx_size=CTX_SIZE):
    """
    each iteration yields a training batch
    """
    df_iter = df.iter_rows()

    while True:
        batch = []

        buf = tokenizer.encode(next(df_iter)[0]).ids
        while len(batch) < batch_size:
            while len(buf) < ctx_size + 1:
                buf += tokenizer.encode("<|endoftext|>").ids
                buf += tokenizer.encode(next(df_iter)[0]).ids

            batch_item = buf[:ctx_size + 1]
            buf = buf[ctx_size + 1:]

            batch += [batch_item]

        yield batch

@TinyJit
def train_step(model: Transformer, batch: Tensor, opt: Optimizer):
    with Tensor.train():
        x = batch[:, :-1]
        y_gt = batch[:, 1:]

        y_hat = model(x)

        loss = y_hat.sparse_categorical_crossentropy(y_gt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        acc_scores = y_hat.argmax(-1) == y_gt
        acc_cnt = acc_scores.sum()

    acc_hit_ids =  acc_scores.where(y_gt, -1).reshape(-1)

    return loss, acc_cnt, len(acc_scores.reshape(-1)), acc_hit_ids

@TinyJit
def eval_step(model, batches):
    loss_mean = Tensor(0.0, requires_grad=False)
    acc_cnt = Tensor(0, requires_grad=False)
    acc_total = len(batches.reshape(-1)) - MINIBATCH_SIZE * N_V_BATCHES

    acc_hit_ids = Tensor.empty(0, CTX_SIZE, dtype=dtypes.int32)

    for b in batches:
        x = b[:, :-1]
        y_gt = b[:, 1:]

        y_hat = model(x)

        loss_mean = loss_mean + y_hat.sparse_categorical_crossentropy(y_gt).item() / len(batches)

        acc_scores = y_hat.argmax(-1) == y_gt
        acc_cnt = acc_cnt + acc_scores.sum()

        acc_hit_ids = acc_hit_ids.cat(acc_scores.where(y_gt, -1), dim=0)


    return loss_mean.numpy(), acc_cnt.numpy(), acc_total, acc_hit_ids.reshape(-1).numpy()

# @TinyJit
def gen_step(model: Transformer, tokenizer: Tokenizer):
    x = sum([tokenizer.encode("<|endoftext|>").ids for _ in range(CTX_SIZE)], [])

    tokens = tokenizer.encode("What").ids

    for i in range(N_GEN_TOKENS):

        for j in range(len(tokens)):
            x[j] = tokens[j]

        ys = model(Tensor(x, requires_grad=False).reshape(1, -1))

        last_y = ys[0][len(tokens) - 1]

        _values, topk = last_y.topk(TOPK_K)

        tok_id = NP_RNG.choice(topk.numpy())

        tokens += [tok_id]

    return tokenizer.decode(x)

if __name__ == "__main__":
    tok = load_tokenizer()

    df = load_dataset()

    df_it = dataset_minibatch_iter(df, tok, MINIBATCH_SIZE)

    v_data_batches = Tensor(list(map(lambda _: next(df_it), range(N_V_BATCHES))), requires_grad=False)

    model = Transformer(CTX_SIZE, NUM_BLOCKS, EMBED_DIM, NUM_HEADS, FF_DIM, tok.get_vocab_size(), DROPOUT)

    params = get_parameters(model)

    param_count = sum(map(lambda t: len(t.reshape(-1)), params))

    print(f"Training a {param_count} parameter model")

    opt_proper = AdamW(params=params, lr=LR)
    opt_warmup = AdamW(params=params, lr=LR * 10)
    lr_sched = CosineAnnealingLR(opt_proper, 100000)

    # I'm a bit tired of setting DEBUG=2 manually when playing with the  metaparameters
    first_pass = True

    ani = animation.FuncAnimation(CHART_DATA.fig, CHART_DATA.plot, interval=1000, blit=False)

    plt.show(block=False)

    opt = opt_warmup
    for i in range(N_BATCHES):
        plt.pause(0.05)

        ctx = None
        if first_pass:
            ctx = Context(DEBUG=2)
        else:
            ctx = Context()

        with ctx:
            if i > WARMUP_ROUNDS:
                opt = opt_proper
                lr_sched.step()
            else:
                print(f"WARMUP {i}/{WARMUP_ROUNDS}")

            batch = Tensor(next(df_it))

            t_loss, t_acc_cnt, t_acc_total, t_acc_hit_ids = train_step(model, batch, opt)

            t_acc = t_acc_cnt / t_acc_total

            t_acc_hit_ids = set(map(int, t_acc_hit_ids.numpy()))
            t_acc_hit_ids.remove(-1)

            t_acc_hit_ids = sorted(list(t_acc_hit_ids))

            t_hits_decoded = tok.decode(t_acc_hit_ids)

            CHART_DATA.t_loss = np.append(CHART_DATA.t_loss, [t_loss.item()])
            CHART_DATA.t_acc = np.append(CHART_DATA.t_acc, [t_acc.item()])
            CHART_DATA.t_acc_unique_hits = np.append(CHART_DATA.t_acc_unique_hits, [len(t_acc_hit_ids)])


            print(f"Step {i+1:10} of {N_BATCHES} | T Loss: {t_loss.item():20.10} | T acc: {t_acc_cnt.item():6} of {t_acc_total:6} ({t_acc.item():20.10}) | T unique acc hits: {len(t_acc_hit_ids):4} {t_acc_hit_ids} {t_hits_decoded}")


            if i % V_INTERVAL == 0:
            
                v_loss, v_acc_cnt, v_acc_total, v_acc_hit_ids = eval_step(model, v_data_batches)

                v_acc = v_acc_cnt / v_acc_total

                v_acc_hit_ids = set(map(int, v_acc_hit_ids))
                v_acc_hit_ids.remove(-1)

                v_acc_hit_ids = sorted(list(t_acc_hit_ids))

                v_hits_decoded = tok.decode(v_acc_hit_ids)

                CHART_DATA.v_loss = np.append(CHART_DATA.v_loss, [v_loss])
                CHART_DATA.v_acc = np.append(CHART_DATA.v_acc, [v_acc])
                CHART_DATA.v_acc_unique_hits = np.append(CHART_DATA.v_acc_unique_hits, [len(v_acc_hit_ids)])

                print(f"Step {i+1:10} of {N_BATCHES} | V Loss: {v_loss:20.10} | V acc: {v_acc_cnt:6} of {v_acc_total:6} ({v_acc:20.10}) | V unique acc hits: {len(v_acc_hit_ids):4} {v_acc_hit_ids} {v_hits_decoded}")

            if i % 500 == 0:
                gen_str = gen_step(model, tok)

                print(f"Gen: {gen_str}")

            sys.stdout.flush()

        first_pass = False
