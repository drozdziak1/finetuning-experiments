from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters

from tokenizers import Tokenizer

import ipdb
import polars

FINEWEB_PATH = "/home/drozdziak1/Documents/datasets/fineweb/000_00000.parquet"

N_BATCHES = 1000000

T_BATCH_SIZE = 50
V_BATCH_SIZE = 20
RNG_SEED = 0xdeadbeef
CTX_SIZE = 100

LR = 0e-5


class Transformer:
    def __init__(self, ctx_size, num_blocks, embed_dim, num_heads, ff_dim, vocab_size):
        assert embed_dim % num_heads == 0

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.ctx_size = ctx_size
        self.vocab_size = vocab_size

        self.embed = Tensor.scaled_uniform(vocab_size, embed_dim)

        self.blocks = [TBlock(embed_dim, num_heads, ff_dim)for _ in range(num_blocks)]

        self.unembed = Tensor.scaled_uniform(embed_dim, vocab_size)

    def __call__(self, x: Tensor):
        """
        X: (B, T)
        """

        x_onehot = x.one_hot(self.vocab_size)

        x_embedded = x_onehot.dot(self.embed)

        x_refined = x_embedded.sequential(self.blocks)

        x_unembed = x_refined.dot(self.unembed)

        return x_unembed

        

        


class TBlock:
    def __init__(self, embed_dim, num_heads, ff_dim):

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim / num_heads
        self.ff_dim = ff_dim
        
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

        x_q, x_k, x_v = [x_ln.dot(t).reshape(x_ln[0], x_ln[1], self.num_heads, self.head_size) for t in [self.q_w, self.k_w, self.v_w]]

        x_atn = Tensor.scaled_dot_product_attention(q, k, v)

        x_atn_out = x_atn.reshape(x.shape).dot(self.atn_out_w)

        x += x_atn_out

        x_ff1 = x.linear(self.ff1_w, self.ff1_b).gelu()

        x_ff2 = x_ff1.linear(self.ff2_w, self.ff2_b)

        x += x_ff2


def load_tokenizer(model_name="gpt2"):
    return Tokenizer.from_pretrained(model_name)

def load_dataset(data_path=FINEWEB_PATH, val_batch_size=V_BATCH_SIZE):
    df = polars.read_parquet(data_path, columns=['text'])

    return df.sample(n=val_batch_size, seed=RNG_SEED), df

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
    x = [sample[:-1] for sample in batch]
    y_gt = [sample[1:] for sample in batch]

    y = model(Tensor(x))

    loss = y.sparse_categorical_crossentropy(y_gt)

    opt.zero_grad()
    
    loss.backward()

    opt.step()

    return loss.numpy()

    

    


if __name__ == "__main__":
    tok = load_tokenizer()

    v_data, t_data = load_dataset()

    t_df_it = dataset_batch_iter(t_data, tok, T_BATCH_SIZE)

    model = Transformer(CTX_SIZE, 16, 64, 4, 256, tok.get_vocab_size())

    opt = AdamW(params=get_parameters(model), lr=LR)

    with Tensor.train():
        for i in range(N_BATCHES):
            loss = train_step(model, next(t_df_it), opt)

            if i % 100 == 0:
                print(f"Step {i-1} | Loss: {loss}")

        
        

    ipdb.set_trace()
    
    
