import time
from collections import defaultdict
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from utils import CfgNode as CN
import random
import math
import time
import nltk
from utils import CfgNode as CN

nltk.download('punkt')
from nltk.tokenize import word_tokenize

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_batches_per_epoch_train = 100
num_batches_per_epoch_valid = 50
BATCH_SIZE = 64
text_file = "coran2.txt"

def dataLoader():
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

text = dataLoader()

def build_vocab(text):
    tokens = word_tokenize(text)
    vocab = sorted(set(tokens))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for i, token in enumerate(vocab)}
    return vocab, token_to_id, id_to_token

vocab, token_to_id, id_to_token = build_vocab(text)
vocab_size = len(vocab)
print(f"Vocab Size: {vocab_size}")

def encode(text):
    tokens = word_tokenize(text)
    token_ids = [token_to_id[token] for token in tokens]
    return token_ids

def decode(token_ids):
    tokens = [id_to_token[token_id] for token_id in token_ids]
    decoded_text = ' '.join(tokens)
    return decoded_text

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, block_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (BATCH_SIZE,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y




def train_iterator(num_batches_per_epoch, block_size):
    while True:
        for _ in range(num_batches_per_epoch):
            yield get_batch("train", block_size)

def valid_iterator(num_batches_per_epoch, block_size):
    while True:
        for _ in range(num_batches_per_epoch):
            yield get_batch("valid", block_size)


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        C.block_size = 64
        C.n_heads = 3
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_iter):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_iter
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = self.train_dataset

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = self.train_dataset
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
def main():
    train_iter = train_iterator(num_batches_per_epoch_train, Trainer.get_default_config().block_size)
    valid_iter = valid_iterator(num_batches_per_epoch_valid, Trainer.get_default_config().block_size)

    from next_word_decoder import GPT

    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = vocab_size
    model_config.block_size = 100
    model = GPT(model_config)
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4
    train_config.max_iters = 20000
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, train_iter)

    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

    trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()

    model_save_path = "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()