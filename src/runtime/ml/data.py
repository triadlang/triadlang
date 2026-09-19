from __future__ import annotations

from runtime.ml.tensor import TriadTensor, tensor
from triad import ntri as np


class Dataset:

    def __init__(self, x, y=None):
        if isinstance(x, TriadTensor):
            x = x._data
        if isinstance(y, TriadTensor):
            y = y._data
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64) if y is not None else None
        if self.y is not None and self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f'Dataset x/y size mismatch: {self.x.shape[0]} vs {self.y.shape[0]}')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            xs = self.x[idx]
            ys = self.y[idx] if self.y is not None else None
            return Dataset(xs, ys)
        if self.y is not None:
            return (tensor(self.x[idx]), tensor(self.y[idx]))
        return tensor(self.x[idx])

    def shuffle(self, seed=None):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self))
        self.x = self.x[perm]
        if self.y is not None:
            self.y = self.y[perm]

    def split(self, ratio=0.8, seed=None):
        rng = np.random.default_rng(seed)
        n = len(self)
        perm = rng.permutation(n)
        split = int(n * ratio)
        train_idx = perm[:split]
        val_idx = perm[split:]
        train_ds = Dataset(self.x[train_idx], self.y[train_idx] if self.y is not None else None)
        val_ds = Dataset(self.x[val_idx], self.y[val_idx] if self.y is not None else None)
        return (train_ds, val_ds)

class DataLoader:

    def __init__(self, dataset: Dataset, batch_size: int=32, shuffle: bool=False, drop_last: bool=False, seed: int | None=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            self.dataset.shuffle(seed=self.seed + self._epoch if self.seed is not None else None)
        n = len(self.dataset)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            if self.drop_last and end - start < self.batch_size:
                break
            xs = self.dataset.x[start:end]
            xs_t = tensor(xs)
            if self.dataset.y is not None:
                ys = self.dataset.y[start:end]
                yield (xs_t, tensor(ys))
            else:
                yield xs_t

    def __next_epoch__(self):
        self._epoch += 1
