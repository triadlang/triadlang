import pytest
import numpy as np
import os
import tempfile
from runtime.tensor import tensor, TriadTensor, mse_loss, no_grad
from runtime.nn import Linear, Sequential, ReLU, Adam, Module
from runtime.data import Dataset, DataLoader
from runtime.trainer import Trainer, EarlyStopping, LRScheduler, LossHistory
from runtime.serialization import save_weights, load_weights, save_checkpoint, load_checkpoint

class TestTrainer:

    def _make_xor_data(self):
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([[0], [1], [1], [0]], dtype=np.float64)
        return Dataset(x, y)

    def test_fit_basic(self):
        np.random.seed(0)
        model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1))
        opt = Adam(model.parameters(), lr=0.05)
        ds = self._make_xor_data()
        dl = DataLoader(ds, batch_size=4)
        trainer = Trainer(model, opt, mse_loss)
        history = trainer.fit(dl, epochs=5, verbose=False)
        assert len(history['train_loss']) == 5
        assert all((isinstance(l, float) for l in history['train_loss']))

    def test_fit_with_validation(self):
        np.random.seed(0)
        model = Linear(3, 1)
        opt = Adam(model.parameters(), lr=0.05)
        x = np.random.randn(40, 3)
        y = x @ np.array([[1.0], [2.0], [-1.0]]) + 0.5
        ds = Dataset(x, y)
        train_ds, val_ds = ds.split(0.8, seed=42)
        train_dl = DataLoader(train_ds, batch_size=8)
        val_dl = DataLoader(val_ds, batch_size=8)
        trainer = Trainer(model, opt, mse_loss)
        history = trainer.fit(train_dl, val_dl, epochs=10, verbose=False)
        assert len(history['train_loss']) == 10
        assert len(history['val_loss']) == 10

    def test_evaluate(self):
        np.random.seed(0)
        model = Linear(2, 1)
        opt = Adam(model.parameters(), lr=0.01)
        ds = Dataset(np.random.randn(10, 2), np.random.randn(10, 1))
        dl = DataLoader(ds, batch_size=5)
        trainer = Trainer(model, opt, mse_loss)
        val_loss = trainer.evaluate(dl)
        assert isinstance(val_loss, float)
        assert val_loss > 0

class TestEarlyStopping:

    def test_triggers(self):
        model = Linear(10, 1)
        opt = Adam(model.parameters(), lr=0.001)
        np.random.seed(0)
        train_x = np.random.randn(20, 10)
        train_y = np.random.randn(20, 1)
        val_x = np.random.randn(20, 10)
        val_y = np.random.randn(20, 1)
        train_dl = DataLoader(Dataset(train_x, train_y), batch_size=20)
        val_dl = DataLoader(Dataset(val_x, val_y), batch_size=20)
        es = EarlyStopping(patience=3, min_delta=0.1)
        trainer = Trainer(model, opt, mse_loss, callbacks=[es])
        trainer.fit(train_dl, val_dl, epochs=50, verbose=False)
        assert es.stopped
        assert len(trainer.history['train_loss']) < 50

    def test_no_trigger_with_improvement(self):
        np.random.seed(42)
        model = Linear(2, 1)
        opt = Adam(model.parameters(), lr=0.05)
        x = np.random.randn(20, 2)
        y = x @ np.array([[1.0], [2.0]])
        ds = Dataset(x, y)
        train_dl = DataLoader(ds, batch_size=10)
        val_dl = DataLoader(ds, batch_size=10)
        es = EarlyStopping(patience=5)
        trainer = Trainer(model, opt, mse_loss, callbacks=[es])
        trainer.fit(train_dl, val_dl, epochs=10, verbose=False)
        assert not es.stopped

class TestLRScheduler:

    def test_reduces_lr(self):
        model = Linear(10, 1)
        opt = Adam(model.parameters(), lr=0.01)
        np.random.seed(0)
        train_x = np.random.randn(20, 10)
        train_y = np.random.randn(20, 1)
        val_x = np.random.randn(20, 10)
        val_y = np.random.randn(20, 1)
        train_dl = DataLoader(Dataset(train_x, train_y), batch_size=20)
        val_dl = DataLoader(Dataset(val_x, val_y), batch_size=20)
        sched = LRScheduler(factor=0.5, patience=2)
        trainer = Trainer(model, opt, mse_loss, callbacks=[sched])
        trainer.fit(train_dl, val_dl, epochs=30, verbose=False)
        assert opt.lr < 0.01

class TestLossHistory:

    def test_records(self):
        model = Linear(2, 1)
        opt = Adam(model.parameters(), lr=0.01)
        ds = Dataset(np.random.randn(10, 2), np.random.randn(10, 1))
        dl = DataLoader(ds, batch_size=5)
        lh = LossHistory()
        trainer = Trainer(model, opt, mse_loss, callbacks=[lh])
        trainer.fit(dl, epochs=3, verbose=False)
        assert len(lh.batch_losses) > 0

class TestSaveLoadWeights:

    def test_round_trip(self):
        model = Linear(4, 3)
        w_before = model.weight._data.copy()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_weights(model, path)
            model.weight._data[:] = 0
            load_weights(model, path)
            assert np.allclose(model.weight._data, w_before)
        finally:
            os.unlink(path)

class TestSaveLoadCheckpoint:

    def test_round_trip(self):
        np.random.seed(42)
        model = Linear(4, 2)
        opt = Adam(model.parameters(), lr=0.01)
        x = tensor(np.random.randn(4, 4))
        y = model(x)
        y.backward(np.ones_like(y._data))
        opt.step()
        w_before = model.weight._data.copy()
        t_before = opt.t
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_checkpoint(model, opt, path, extra={'epoch': 5})
            model.weight._data[:] = 0
            opt.t = 0
            extra = load_checkpoint(model, opt, path)
            assert np.allclose(model.weight._data, w_before)
            assert opt.t == t_before
            assert extra['epoch'] == 5
        finally:
            os.unlink(path)

class TestEndToEndTraining:

    def test_linear_regression_with_trainer(self):
        np.random.seed(42)
        model = Linear(3, 1, bias=True)
        opt = Adam(model.parameters(), lr=0.05)
        x_data = np.random.randn(50, 3)
        true_w = np.array([[1.0], [2.0], [-1.0]])
        y_data = x_data @ true_w + 0.5
        ds = Dataset(x_data, y_data)
        train_ds, val_ds = ds.split(0.8, seed=42)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=16)
        trainer = Trainer(model, opt, mse_loss)
        trainer.fit(train_dl, val_dl, epochs=100, verbose=False)
        with no_grad():
            x_test = tensor(x_data)
            y_test = tensor(y_data)
            pred = model(x_test)
            final_loss = mse_loss(pred, y_test)
            assert final_loss._data < 0.5