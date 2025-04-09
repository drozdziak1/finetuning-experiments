from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW, Optimizer
from tinygrad.nn.state import get_parameters
from tinygrad.engine.jit import TinyJit

from tinygrad import Context, Device
from tinygrad.nn import LayerNorm, Linear, Embedding
from tokenizers import Tokenizer
from typing import List

import ipdb
import numpy as np
import polars
import random
import sys


FINEWEB_PATH = "/home/drozdziak1/Documents/datasets/fineweb/000_00000.parquet"

N_BATCHES = 16384 * 16384

N_V_BATCHES = 1

BATCH_SIZE = 16

MINIBATCH_SIZE = 4
RNG_SEED = 0xfacebeef
random.seed(RNG_SEED)

Tensor.manual_seed(RNG_SEED)

CTX_SIZE = 128
NUM_BLOCKS = 8
EMBED_DIM = 64
NUM_HEADS = 8
FF_DIM = 128
DROPOUT = 0.1

LR = 3e-5

TOPK_K = 2

# Also stop training when train loss / val loss reaches this value
TARGET_TV_LOSS_RATIO = 0.001

def gen_pos_enc(ctx_size, embed_dim):
    pe = np.zeros((ctx_size, embed_dim))

    pos = np.arange(0, ctx_size, dtype=np.float32).reshape(-1, 1)

    div_term = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))

    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)

    return Tensor(pe, requires_grad=False)


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

        self.pe = gen_pos_enc(ctx_size, embed_dim)

        self.blocks = [TBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)]

        self.unembed = Tensor.kaiming_normal(embed_dim, vocab_size)

    def __call__(self, x: Tensor):
        """
        X: (B, T)
        """

        x_embedded = self.embed(x)

        batch_size, seq_len = x.shape

        position_embeddings = self.pe[:seq_len].unsqueeze(0) # (1, seq_len, embedding_dim)

        x_pe = x_embedded + position_embeddings.expand(batch_size, -1, -1)

        x_refined = x_pe.sequential(self.blocks)

        x_unembed = x_refined.dot(self.unembed).softmax()

        return x_unembed






class TBlock:
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        # self.ln = LayerNorm(embed_dim)

        self.q_w = Tensor.kaiming_normal(embed_dim, embed_dim)
        self.k_w = Tensor.kaiming_normal(embed_dim, embed_dim)
        self.v_w = Tensor.kaiming_normal(embed_dim, embed_dim)

        self.atn_out_w = Tensor.kaiming_normal(embed_dim, embed_dim)

        self.ff1 = Linear(embed_dim, ff_dim)
        self.ff2 = Linear(ff_dim, embed_dim)


    def __call__(self, x: Tensor):
        """
        X: (B, T, C)
        """

        x_ln = x

        x_q, x_k, x_v = [x_ln.dot(t).reshape(x_ln.shape[0], x_ln.shape[1], self.num_heads, self.head_size) for t in [self.q_w, self.k_w, self.v_w]]

        x_atn = Tensor.scaled_dot_product_attention(x_q, x_k, x_v, is_causal=True)

        x_atn_out = x_atn.reshape(x.shape).dot(self.atn_out_w)

        x = x + x_atn_out.dropout(self.dropout)

        x_ff1 = self.ff1(x).gelu()

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
def train_step(model: Transformer, batch: List[List[List[int]]], opt: Optimizer):
    loss_mean = Tensor(0.0)
    acc = Tensor(0.0)
    with Tensor.train():

        opt.zero_grad()

        for mb in batch:
            x = [sample[:-1] for sample in mb]
            y_gt = [sample[1:] for sample in mb]

            y_hat = model(Tensor(x))

            loss = y_hat.sparse_categorical_crossentropy(Tensor(y_gt)).backward()

            loss_mean = loss_mean + loss / len(batch)

            acc_scores = y_hat.argmax(-1) == Tensor(y_gt)
            acc = acc + acc_scores.mean() / len(batch)

        opt.step()

    return loss_mean.numpy(), acc.numpy()

@TinyJit
def eval_step(model, batches):
    loss_mean = Tensor(0.0)
    acc = Tensor(0.0)

    for b in batches:
        x = [sample[:-1] for sample in b]
        y_gt = [sample[1:] for sample in b]

        y_hat = model(Tensor(x))

        loss_mean = loss_mean + y_hat.sparse_categorical_crossentropy(Tensor(y_gt)).item() / len(batches)

        acc_scores = y_hat.argmax(-1) == Tensor(y_gt)
        acc = acc + acc_scores.mean() / len(batch)

    return loss_mean.numpy(), acc.numpy()

# @TinyJit
def gen_step(model: Transformer, tokenizer: Tokenizer):
    x = sum([tokenizer.encode("<|endoftext|>").ids for _ in range(CTX_SIZE)], [])

    k = TOPK_K

    tokens = tokenizer.encode("What").ids

    for i in range(20):

        for j in range(len(tokens)):
            x[j] = tokens[j]

        ys = model(Tensor(x).reshape(1, -1))

        last_y = ys[0][len(tokens) - 1].numpy()

        topk = last_y.argpartition(-k)[-k:]

        tok_id = np.random.choice(topk)

        tokens += [tok_id]

    return tokenizer.decode(x)

if __name__ == "__main__":
    tok = load_tokenizer()

    df = load_dataset()

    df_it = dataset_minibatch_iter(df, tok, MINIBATCH_SIZE)

    v_data_batches = list(map(lambda _: next(df_it), range(N_V_BATCHES)))

    model = Transformer(CTX_SIZE, NUM_BLOCKS, EMBED_DIM, NUM_HEADS, FF_DIM, tok.get_vocab_size(), DROPOUT)

    params = get_parameters(model)

    param_count = sum(map(lambda t: len(t.reshape(-1)), params))

    print(f"Training a {param_count} parameter model")

    opt = AdamW(params=params, lr=LR)

    # I'm a bit tired of setting DEBUG=2 manually when playing with the  metaparameters
    first_pass = True

    for i in range(N_BATCHES):
        ctx = None
        if first_pass:
            ctx = Context(DEBUG=2)
        else:
            ctx = Context()

        with ctx:
            batch = [next(df_it) for _ in range(BATCH_SIZE)]

            t_loss, t_acc = train_step(model, batch, opt)

            print(f"Step {i+1:10} | T Loss: {t_loss:10.5} | T acc: {t_acc:7.5}")

            if i % 20 == 0:
                v_loss, v_acc = eval_step(model, v_data_batches)

                print(f"Step {i+1:10} | V Loss: {v_loss:10.5} | V acc: {v_acc:7.5}")

                if t_loss / v_loss < TARGET_TV_LOSS_RATIO:
                    print(f"Target T/V loss ratio {TARGET_TV_LOSS_RATIO} reached")
                    break

            if i % 100 == 0:
                gen_str = gen_step(model, tok)

                print(f"Gen: {gen_str}")

        first_pass = False
