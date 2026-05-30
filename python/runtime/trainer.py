from __future__ import annotations
import time
from runtime.tensor import TriadTensor, no_grad
from runtime.data import DataLoader
from runtime.nn import Module

class Trainer:

    def __init__(self, model: Module, optimizer, loss_fn, metrics: dict | None=None, callbacks: list | None=None, device: str='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or {}
        self.callbacks = callbacks or []
        self.device = device
        self.history: dict[str, list] = {'train_loss': [], 'val_loss': []}

    def _run_callbacks(self, event: str, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, event, None)
            if fn:
                fn(trainer=self, **kwargs)

    def _compute_metrics(self, pred: TriadTensor, target: TriadTensor) -> dict[str, float]:
        results = {}
        for name, fn in self.metrics.items():
            results[name] = fn(pred, target)
        return results

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.training = True if hasattr(self.model, 'training') else None
        total_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            if isinstance(batch, tuple):
                x, y = batch
            else:
                continue
            self._run_callbacks('on_batch_start', batch_idx=n_batches)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss._data)
            n_batches += 1
            self._run_callbacks('on_batch_end', batch_idx=n_batches, loss=float(loss._data))
        if hasattr(dataloader, '__next_epoch__'):
            dataloader.__next_epoch__()
        return total_loss / max(n_batches, 1)

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.training = False if hasattr(self.model, 'training') else None
        total_loss = 0.0
        n_batches = 0
        with no_grad():
            for batch in dataloader:
                if isinstance(batch, tuple):
                    x, y = batch
                else:
                    continue
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                total_loss += float(loss._data)
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None=None, epochs: int=10, verbose: bool=True) -> dict[str, list]:
        self._run_callbacks('on_train_start')
        self._stop_requested = False
        for epoch in range(epochs):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            val_loss = None
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
            elapsed = time.time() - t0
            self._run_callbacks('on_epoch_end', epoch=epoch, train_loss=train_loss, val_loss=val_loss, elapsed=elapsed)
            if verbose:
                msg = f'Epoch {epoch + 1}/{epochs} — loss: {train_loss:.4f}'
                if val_loss is not None:
                    msg += f' — val_loss: {val_loss:.4f}'
                msg += f' ({elapsed:.2f}s)'
                print(msg)
            if self._stop_requested:
                break
        self._run_callbacks('on_train_end')
        return self.history

class EarlyStopping:

    def __init__(self, patience: int=5, min_delta: float=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stopped = False

    def on_epoch_end(self, trainer: Trainer, **kwargs):
        val_loss = kwargs.get('val_loss')
        if val_loss is None:
            return
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                trainer._stop_requested = True

class LRScheduler:

    def __init__(self, factor: float=0.5, patience: int=3):
        self.factor = factor
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def on_epoch_end(self, trainer: Trainer, **kwargs):
        val_loss = kwargs.get('val_loss')
        if val_loss is None:
            return
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.optimizer.lr *= self.factor
                self.counter = 0

class LossHistory:

    def __init__(self):
        self.batch_losses: list[float] = []

    def on_batch_end(self, **kwargs):
        loss = kwargs.get('loss')
        if loss is not None:
            self.batch_losses.append(loss)