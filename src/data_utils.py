import datasets

import multiprocessing as mp
import sys

from tokenizers import Tokenizer
from multiprocessing import Process

from my_globals import CFG

def load_tokenizer(model_name="gpt2"):
    return Tokenizer.from_pretrained(model_name)

def load_dataset(ds_quick: bool, rng_seed: int):
    if ds_quick:
        ds = datasets.load_dataset("HuggingFaceFW/fineweb", data_files="sample/100BT/000_00000.parquet", split="train", )
    else:
        ds = datasets.load_dataset("HuggingFaceFW/fineweb", "sample-100BT", split="train")

    ds_iter = ds.to_iterable_dataset().shuffle(buffer_size=10_000, seed=rng_seed).iter(1)

    return ds_iter

def dataset_batch_iter(ds_iter, tokenizer, batch_size, minibatch_size, ctx_size):
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

class NBBatchQueue:
    def __init__(self, generator, max_q_size, q_poll_interval=0.1):
        self.q = mp.Queue(maxsize=max_q_size)
        self.q_poll_interval = q_poll_interval
        self.generator = generator
        self.prod_task = Process(target=self.next_item_mp_target, name="AsyncBatchQueue-prod-task", daemon=True)
        self.prod_task.start()

    def next_item_mp_target(self):
        try:
            for item in self.generator:
                while True:
                    try:
                        self.q.put(item, timeout=self.q_poll_interval)
                        break
                    except mp.queues.Full as full:
                        pass

                    # We do this here to make sure this process is not left hanging
                    if CFG.quitting:
                        print("AsyncBatchQueue: Bye!")
                        sys.exit(0)

        except Exception as e:
            print(f"Async batch queue encountered an error: {e}")

        print("Async batch queue exhausted, wrapping up...")

        sys.stdout.flush()
        self.q.put(None, block=True)

    def next_item(self, debug=False):
        item = self.q.get(block=True)
        if debug:
            print(f"Getting item {item} from queue")

        return item

    def __del__(self):
        if self.prod_task.is_alive():
            self.prod_task.kill()
