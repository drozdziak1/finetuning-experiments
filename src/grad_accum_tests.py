from tinygrad import Tensor, Context, TinyJit
from tinygrad.nn import Linear
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters

BATCH_SIZE = 10_000

MINIBATCH_SIZE = 10_000
N_CLASSES = 10

N_ITERS = 1_000

class Model:
    def __init__(self):
        self.l1 = Linear(N_CLASSES, 4 * N_CLASSES)
        self.l2 = Linear(4 * N_CLASSES, 16 * N_CLASSES)
        self.l3 = Linear(16 * N_CLASSES, 4 * N_CLASSES)
        self.l4 = Linear(4 * N_CLASSES, N_CLASSES)


    def __call__(self, x: Tensor) -> Tensor:
        x_l1 = self.l1(x)
        x_l2 = self.l2(x_l1)
        x_l3 = self.l3(x_l2)
        x_l4 = self.l4(x_l3)

        return x_l4.softmax(axis=-1)

def gen_train_data():
    x = Tensor.randn(MINIBATCH_SIZE,N_CLASSES)
    y = x.argmin(-1)

    return x, y

def epoch_step(model, opt, ctx):
    @TinyJit
    def mb_step(model):
        x, y_gt = gen_train_data()

        y_hat = model(x)

        loss = y_hat.sparse_categorical_crossentropy(y_gt).backward()
        acc = (y_hat.argmax(-1) == y_gt).mean()
        acc.requires_grad = False

        return loss, acc

    with ctx:
        opt.zero_grad()

        loss_mean = 0.0
        acc_mean = 0.0

        for j in range(BATCH_SIZE):
            loss, acc = mb_step(model)

            loss_mean += loss.item() / BATCH_SIZE
            acc_mean += acc.item() / BATCH_SIZE



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

        print(f"Loss: {loss_mean:10.5} Acc: {acc_mean:10.5}")

        first_iter = False





if __name__ == "__main__":
    main()
