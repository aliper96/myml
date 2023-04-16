import numpy as np


def get_batch(data, block_size, batch_size):
    ix = np.random.randint(len(data) - block_size, size=batch_size)
    print(ix)
    x = np.stack([np.array(data[i:i + block_size]) for i in ix])
    y = np.stack([np.array(data[i + 1:i + 1 + block_size]) for i in ix])

    return x, y


data = [i for i in range(100)]

block_size = 10
batch_size = 5
for i in range(100000):
    x, y = get_batch(data, block_size, batch_size)

