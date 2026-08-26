from __future__ import annotations

import time

from runtime.ml.data import DataLoader
from runtime.ml.nn import Module
from runtime.ml.tensor import TriadTensor, no_grad
from triad import ntri as np


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
        scaler = self._find_scaler()
        for batch in dataloader:
            if isinstance(batch, tuple):
                x, y = batch
            else:
                continue
            self._run_callbacks('on_batch_start', batch_idx=n_batches)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            if scaler is not None:
                scaled = scaler.scale(loss)
                scaled.backward()
            else:
                loss.backward()
            if scaler is not None:
                if not scaler.step(self.optimizer):
                    scaler.update()
                    self._run_callbacks('on_batch_end', batch_idx=n_batches, loss=float(loss._data))
                    n_batches += 1
                    continue
                scaler.update()
            else:
                self.optimizer.step()
            total_loss += float(loss._data)
            n_batches += 1
            self._run_callbacks('on_batch_end', batch_idx=n_batches, loss=float(loss._data))
        if hasattr(dataloader, '__next_epoch__'):
            dataloader.__next_epoch__()
        return total_loss / max(n_batches, 1)

    def _find_scaler(self):
        for cb in self.callbacks:
            scaler = getattr(cb, '_scaler', None)
            if scaler is not None and getattr(cb, 'enabled', False):
                return scaler
        return None

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

class CosineAnnealingLR:

    def __init__(self, eta_min: float = 0.0, T_max: int | None = None):
        self.eta_min = eta_min
        self.T_max = T_max
        self.base_lr = None

    def on_train_start(self, trainer: Trainer, **kwargs):
        self.base_lr = trainer.optimizer.lr

    def on_epoch_end(self, trainer: Trainer, **kwargs):
        epoch = kwargs.get('epoch', 0)
        epochs_run = epoch + 1
        T = self.T_max or kwargs.get('total_epochs', epochs_run)
        lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * epochs_run / T))
        trainer.optimizer.lr = lr

class WarmupCosineScheduler:

    def __init__(self, warmup_epochs: int = 5, eta_min: float = 0.0, T_max: int | None = None):
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.T_max = T_max
        self.base_lr = None

    def on_train_start(self, trainer: Trainer, **kwargs):
        self.base_lr = trainer.optimizer.lr

    def on_epoch_end(self, trainer: Trainer, **kwargs):
        epoch = kwargs.get('epoch', 0)
        total = kwargs.get('total_epochs', self.T_max or (epoch + 1))
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(total - self.warmup_epochs, 1)
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * progress))
        trainer.optimizer.lr = lr

class OneCycleLR:

    def __init__(self, max_lr: float = 0.01, div_factor: float = 25.0,
                 final_div_factor: float = 10000.0, pct_start: float = 0.3):
        self.max_lr = max_lr
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.pct_start = pct_start

    def on_train_start(self, trainer: Trainer, **kwargs):
        trainer.optimizer.lr = self.initial_lr

    def on_epoch_end(self, trainer: Trainer, **kwargs):
        epoch = kwargs.get('epoch', 0)
        total = kwargs.get('total_epochs', epoch + 1)
        pct = (epoch + 1) / total
        if pct <= self.pct_start:
            scale = pct / self.pct_start
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * scale
        else:
            scale = (pct - self.pct_start) / (1.0 - self.pct_start)
            lr = self.max_lr - (self.max_lr - self.final_lr) * scale
        trainer.optimizer.lr = lr

class GradientClipping:

    def __init__(self, max_norm: float = 1.0, clip_type: str = 'norm'):
        self.max_norm = max_norm
        self.clip_type = clip_type

    def on_batch_start(self, **kwargs):
        pass

    def on_batch_end(self, trainer: Trainer, **kwargs):
        params = trainer.model.parameters()
        if self.clip_type == 'norm':
            total_norm_sq = 0.0
            for p in params:
                if p._grad is not None:
                    total_norm_sq += np.sum(p._grad ** 2)
            total_norm = np.sqrt(total_norm_sq)
            if total_norm > self.max_norm:
                scale = self.max_norm / (total_norm + 1e-6)
                for p in params:
                    if p._grad is not None:
                        p._grad = p._grad * scale
        elif self.clip_type == 'value':
            for p in params:
                if p._grad is not None:
                    p._grad = np.clip(p._grad, -self.max_norm, self.max_norm)

class GradientAccumulation:

    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self._step_count = 0
        self._original_step = None

    def on_train_start(self, trainer: Trainer, **kwargs):
        self._original_step = trainer.optimizer.step
        self._step_count = 0
        trainer._accumulation_active = True

    def on_batch_end(self, trainer: Trainer, **kwargs):
        self._step_count += 1
        if self._step_count % self.accumulation_steps == 0:
            for p in trainer.model.parameters():
                if p._grad is not None:
                    p._grad = p._grad / self.accumulation_steps
            self._original_step()
            trainer.optimizer.zero_grad()

class MixedPrecision:

    def __init__(self, enabled: bool = True, init_scale: float = 2.0 ** 16):
        self.enabled = enabled
        self._scaler = None

    def on_train_start(self, trainer: Trainer, **kwargs):
        if self.enabled:
            from runtime.ml.amp import GradScaler
            self._scaler = GradScaler(init_scale=2.0 ** 16)

    def on_batch_start(self, trainer: Trainer, **kwargs):
        if not self.enabled or self._scaler is None:
            return
        from runtime.ml.amp import autocast
        trainer._amp_ctx = autocast()
        trainer._amp_ctx.__enter__()

    def on_batch_end(self, trainer: Trainer, **kwargs):
        if not self.enabled or self._scaler is None:
            return
        amp_ctx = getattr(trainer, '_amp_ctx', None)
        if amp_ctx is not None:
            amp_ctx.__exit__(None, None, None)
            trainer._amp_ctx = None

class SWA:

    def __init__(self, start_epoch: int = 10, swa_lr: float = 0.05, avg_fn: str = 'ema'):
        self.start_epoch = start_epoch
        self.swa_lr = swa_lr
        self.avg_fn = avg_fn
        self._swa_params = None
        self._n_averaged = 0

    def on_train_start(self, trainer: Trainer, **kwargs):
        self._swa_params = None
        self._n_averaged = 0

    def on_epoch_end(self, trainer: Trainer, **kwargs):
        epoch = kwargs.get('epoch', 0)
        if epoch < self.start_epoch:
            return
        trainer.optimizer.lr = self.swa_lr
        params = trainer.model.parameters()
        if self._swa_params is None:
            self._swa_params = [p._data.copy() for p in params]
        else:
            self._n_averaged += 1
            for i, p in enumerate(params):
                if self.avg_fn == 'ema':
                    self._swa_params[i] = 0.9 * self._swa_params[i] + 0.1 * p._data
                else:
                    self._swa_params[i] = (self._swa_params[i] * self._n_averaged + p._data) / (self._n_averaged + 1)

    def on_train_end(self, trainer: Trainer, **kwargs):
        if self._swa_params is not None:
            for i, p in enumerate(trainer.model.parameters()):
                p._data = self._swa_params[i].copy()

class ModelCheckpoint:

    def __init__(self, monitor: str = 'val_loss', mode: str = 'min',
                 save_path: str = 'best_model.bin', verbose: bool = True):
        self.monitor = monitor
        self.mode = mode
        self.save_path = save_path
        self.verbose = verbose
        self.best = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, trainer: Trainer, **kwargs):
        value = kwargs.get(self.monitor)
        if value is None:
            return
        improved = (value < self.best) if self.mode == 'min' else (value > self.best)
        if improved:
            self.best = value
            from runtime.ml.serialization import save_weights
            save_weights(trainer.model, self.save_path)
            if self.verbose:
                print(f'  checkpoint saved ({self.monitor}={value:.4f})')

class LambdaScheduler:

    def __init__(self, lr_lambda):
        self.lr_lambda = lr_lambda
        self.base_lr = None

    def on_train_start(self, trainer: Trainer, **kwargs):
        self.base_lr = trainer.optimizer.lr

    def on_epoch_end(self, trainer: Trainer, **kwargs):
        epoch = kwargs.get('epoch', 0)
        trainer.optimizer.lr = self.base_lr * self.lr_lambda(epoch)
