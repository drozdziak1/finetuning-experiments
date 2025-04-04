from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from tinygrad.engine.jit import TinyJit

from tinygrad import Context, Device

from tokenizers import Tokenizer

import ipdb
import polars

FINEWEB_PATH = "/home/drozdziak1/Documents/datasets/fineweb/000_00000.parquet"

N_BATCHES = 4096

BATCH_SIZE = 8
RNG_SEED = 0xdeadbeef

CTX_SIZE = 4096
NUM_BLOCKS = 8 EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
DROPOUT = 0.1

LR = 3e-5

# Also stop training when train loss / val loss reaches this value
TARGET_TV_LOSS_RATIO = 0.9


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

        self.embed = Tensor.scaled_uniform(vocab_size, embed_dim)

        self.blocks = [TBlock(embed_dim, num_heads, ff_dim, dropout)for _ in range(num_blocks)]

        self.unembed = Tensor.scaled_uniform(embed_dim, vocab_size)

    def __call__(self, x: Tensor):
        """
        X: (B, T)
        """

        x_onehot = x.one_hot(self.vocab_size)

        x_embedded = x_onehot.dot(self.embed)

        x_refined = x_embedded.sequential(self.blocks)

        x_unembed = x_refined.dot(self.unembed).log_softmax()

        return x_unembed

        

        


class TBlock:
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        self.ln_w = Tensor.ones(embed_dim)
        self.ln_b = Tensor.zeros(embed_dim)


        self.q_w = Tensor.scaled_uniform(embed_dim, embed_dim)
        self.k_w = Tensor.scaled_uniform(embed_dim, embed_dim)
        self.v_w = Tensor.scaled_uniform(embed_dim, embed_dim)

        self.atn_out_w = Tensor.scaled_uniform(embed_dim, embed_dim)
        
        self.ff1_w = Tensor.scaled_uniform(embed_dim, ff_dim)
        self.ff1_b = Tensor.scaled_uniform(ff_dim)

        self.ff2_w = Tensor.scaled_uniform(ff_dim, embed_dim)
        self.ff2_b = Tensor.scaled_uniform(embed_dim)
        
    def __call__(self, x: Tensor):
        """
        X: (B, T, C)
        """

        x_ln = x.layernorm().linear(self.ln_w, self.ln_b)

        x_q, x_k, x_v = [x_ln.dot(t).reshape(x_ln.shape[0], x_ln.shape[1], self.num_heads, self.head_size) for t in [self.q_w, self.k_w, self.v_w]]

        x_atn = Tensor.scaled_dot_product_attention(x_q, x_k, x_v)

        x_atn_out = x_atn.reshape(x.shape).dot(self.atn_out_w)

        x = x + x_atn_out.dropout(self.dropout)

        x_ff1 = x.linear(self.ff1_w, self.ff1_b).gelu()

        x_ff2 = x_ff1.linear(self.ff2_w, self.ff2_b)

        x = x + x_ff2.dropout(self.dropout)

        return x


def load_tokenizer(model_name="gpt2"):
    return Tokenizer.from_pretrained(model_name)

def load_dataset(data_path=FINEWEB_PATH):
    df = polars.read_parquet(data_path, columns=['text'])

    return df

def dataset_batch_iter(df, tokenizer, batch_size, ctx_size=CTX_SIZE):
    """
    each iteration yields a training batch
    """
    df_iter = df.iter_rows()

    while True:
        batch = []

        buf = tokenizer.encode(next(df_iter)[0]).ids
        while len(batch) < batch_size:
            while len(buf) < ctx_size + 1:
                buf += tokenizer.encode("<|end_of_text|>").ids
                buf += tokenizer.encode(next(df_iter)[0]).ids

            batch_item = buf[:ctx_size + 1]
            buf = buf[ctx_size + 1:]

            batch += [batch_item]

        yield batch


def train_step(model, batch, opt):
    with Tensor.train():
        x = [sample[:-1] for sample in batch]
        y_gt = [sample[1:] for sample in batch]

        y = model(Tensor(x))

        loss = y.sparse_categorical_crossentropy(Tensor(y_gt))

        opt.zero_grad()

        loss.backward()

        opt.step()

        return loss.numpy()

train_step = TinyJit(train_step)

def eval_step(model, batch):
    x = [sample[:-1] for sample in batch]
    y_gt = [sample[1:] for sample in batch]

    y = model(Tensor(x))

    loss = y.sparse_categorical_crossentropy(Tensor(y_gt))

    return loss.numpy()
    
eval_step = TinyJit(eval_step)
    


if __name__ == "__main__":
    tok = load_tokenizer()

    df = load_dataset()

    df_it = dataset_batch_iter(df, tok, BATCH_SIZE)

    v_data_batch = next(df_it)

    model = Transformer(CTX_SIZE, NUM_BLOCKS, EMBED_DIM, NUM_HEADS, FF_DIM, tok.get_vocab_size(), DROPOUT)

    params = get_parameters(model)

    param_count = sum(map(lambda t: len(t.reshape(-1)), params))

    print(f"Training a {param_count} parameter model")

    opt = AdamW(params=params, lr=LR)

    for i in range(N_BATCHES):
        t_loss = train_step(model, next(df_it), opt)

        print(f"Step {i+1:10} | T Loss: {t_loss}")

        if i % 20 == 0:
            v_loss = eval_step(model, v_data_batch)

            print(f"Step {i+1:10} | V Loss: {v_loss}")

            if t_loss / v_loss < TARGET_TV_LOSS_RATIO:
                print(f"Reached target t/v loss ratio {TARGET_TV_LOSS_RATIO}")
                break


            
