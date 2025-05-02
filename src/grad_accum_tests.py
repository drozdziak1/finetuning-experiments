from tinygrad import Tensor, Context, TinyJit
from tinygrad.nn import Linear
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters

BATCH_SIZE = 100

MINIBATCH_SIZE = 10_000
N_CLASSES = 10

N_ITERS = 1_000

class Model:
    def __init__(self):
        self.l1 = Linear(N_CLASSES, 4 * N_CLASSES)
        self.l2 = Linear(4 * N_CLASSES, 16 * N_CLASSES)
        self.l3 = Linear(16 * N_CLASSES, 4 * N_CLASSES)
        self.l4 = Linear(4 * N_CLASSES, N_CLASSES)


    def __call__(self, x: Tensor):
        x_l1 = self.l1(x)
        x_l2 = self.l2(x_l1)
        x_l3 = self.l3(x_l2)
        x_l4 = self.l4(x_l3)

        return x_l4.softmax(axis=-1)

def gen_train_data():
    x = Tensor.randn(MINIBATCH_SIZE,N_CLASSES)
    y = x.argmin(-1)

    return x, y

@TinyJit
def epoch_step(model, opt, ctx):
        with ctx:
            opt.zero_grad()

            loss_mean = Tensor(0.0, requires_grad=False)
            acc_mean = Tensor(0.0, requires_grad=False)

            for j in range(BATCH_SIZE):
                x, y_gt = gen_train_data()

                y_hat = model(x)

                loss = y_hat.sparse_categorical_crossentropy(y_gt).backward()
                acc = (y_hat.argmax(-1) == y_gt).mean()

                loss_mean = loss_mean + (loss / BATCH_SIZE)
                acc_mean = acc_mean + (acc / BATCH_SIZE)

            opt.step()

            return loss_mean, acc_mean

def main():
    model = Model()
    opt = AdamW(params=get_parameters(model), lr=3e-4)

    Tensor.training = True

    first_iter = True

    for i in range(N_ITERS):
        ctx = Context(DEBUG=2) if first_iter else Context()

        loss_mean, acc_mean = epoch_step(model, opt, ctx)

        print(f"Loss: {loss_mean.item():10.5} Acc: {acc_mean.item():10.5}")

        first_iter = False

    
    
    

if __name__ == "__main__":
    main()
