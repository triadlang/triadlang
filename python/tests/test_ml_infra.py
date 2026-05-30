import pytest
import numpy as np
from runtime.tensor import tensor, TriadTensor
from runtime.data import Dataset, DataLoader
from runtime.losses import bce_loss, binary_cross_entropy_with_logits, huber_loss, kl_div, cosine_similarity_loss, l1_loss, smooth_l1_loss, nll_loss
from runtime.metrics import accuracy, top_k_accuracy, perplexity, confusion_matrix, precision_recall_f1, r2_score, mean_absolute_error, root_mean_squared_error

class TestDataset:

    def test_create(self):
        ds = Dataset(np.random.randn(10, 3), np.random.randn(10, 1))
        assert len(ds) == 10

    def test_getitem(self):
        ds = Dataset(np.array([[1, 2], [3, 4]]), np.array([[5], [6]]))
        x, y = ds[0]
        assert isinstance(x, TriadTensor)
        assert isinstance(y, TriadTensor)

    def test_slice(self):
        ds = Dataset(np.random.randn(10, 3), np.random.randn(10, 1))
        sub = ds[0:5]
        assert isinstance(sub, Dataset)
        assert len(sub) == 5

    def test_no_labels(self):
        ds = Dataset(np.random.randn(10, 3))
        assert len(ds) == 10
        x = ds[0]
        assert isinstance(x, TriadTensor)

    def test_from_tensors(self):
        x = tensor(np.random.randn(5, 2))
        y = tensor(np.random.randn(5, 1))
        ds = Dataset(x, y)
        assert len(ds) == 5

    def test_shuffle(self):
        ds = Dataset(np.arange(10).reshape(10, 1).astype(float))
        ds.shuffle(seed=42)
        assert not np.allclose(ds.x, np.arange(10).reshape(10, 1).astype(float))

    def test_split(self):
        ds = Dataset(np.random.randn(100, 3), np.random.randn(100, 1))
        train, val = ds.split(0.8, seed=42)
        assert len(train) == 80
        assert len(val) == 20

class TestDataLoader:

    def test_batch_count(self):
        ds = Dataset(np.random.randn(10, 3))
        dl = DataLoader(ds, batch_size=3)
        assert len(dl) == 4

    def test_batch_count_drop_last(self):
        ds = Dataset(np.random.randn(10, 3))
        dl = DataLoader(ds, batch_size=3, drop_last=True)
        assert len(dl) == 3

    def test_iter_batches(self):
        ds = Dataset(np.random.randn(10, 3), np.random.randn(10, 1))
        dl = DataLoader(ds, batch_size=4)
        batches = list(dl)
        assert len(batches) == 3
        assert batches[0][0].shape == (4, 3)
        assert batches[-1][0].shape == (2, 3)

    def test_iter_unlabeled(self):
        ds = Dataset(np.random.randn(10, 3))
        dl = DataLoader(ds, batch_size=5)
        batches = list(dl)
        assert len(batches) == 2
        assert isinstance(batches[0], TriadTensor)

    def test_shuffle(self):
        ds = Dataset(np.arange(10).reshape(10, 1).astype(float))
        dl = DataLoader(ds, batch_size=5, shuffle=True, seed=42)
        batches = list(dl)
        assert len(batches) == 2

    def test_full_batch(self):
        ds = Dataset(np.random.randn(5, 3))
        dl = DataLoader(ds, batch_size=5)
        batches = list(dl)
        assert len(batches) == 1
        assert batches[0].shape == (5, 3)

class TestBCELoss:

    def test_perfect(self):
        pred = tensor(np.array([1.0, 0.0, 1.0]))
        target = tensor(np.array([1.0, 0.0, 1.0]))
        loss = bce_loss(pred, target)
        assert loss._data < 0.01

    def test_backward(self):
        pred = tensor(np.array([0.8, 0.2]), requires_grad=True)
        target = tensor(np.array([1.0, 0.0]))
        loss = bce_loss(pred, target)
        loss.backward()
        assert pred._grad is not None

class TestBCEWithLogits:

    def test_basic(self):
        logits = tensor(np.array([2.0, -2.0]))
        target = tensor(np.array([1.0, 0.0]))
        loss = binary_cross_entropy_with_logits(logits, target)
        assert loss._data > 0

    def test_backward(self):
        logits = tensor(np.array([1.0, -1.0]), requires_grad=True)
        target = tensor(np.array([1.0, 0.0]))
        loss = binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        assert logits._grad is not None

class TestHuberLoss:

    def test_small_diff(self):
        pred = tensor(np.array([1.1, 2.05]))
        target = tensor(np.array([1.0, 2.0]))
        loss = huber_loss(pred, target, delta=1.0)
        assert loss._data < 0.01

    def test_large_diff(self):
        pred = tensor(np.array([5.0]))
        target = tensor(np.array([1.0]))
        loss = huber_loss(pred, target, delta=1.0)
        assert loss._data > 1.0

    def test_backward(self):
        pred = tensor(np.array([3.0, 1.0]), requires_grad=True)
        target = tensor(np.array([1.0, 1.0]))
        loss = huber_loss(pred, target)
        loss.backward()
        assert pred._grad is not None

class TestKLDiv:

    def test_identical(self):
        p = tensor(np.array([[np.log(0.5), np.log(0.5)]]))
        q = tensor(np.array([[0.5, 0.5]]))
        loss = kl_div(p, q)
        assert loss._data < 1e-06

class TestCosineSimilarityLoss:

    def test_identical(self):
        a = tensor(np.array([[1.0, 0.0]]))
        b = tensor(np.array([[1.0, 0.0]]))
        loss = cosine_similarity_loss(a, b)
        assert loss._data < 1e-06

    def test_opposite(self):
        a = tensor(np.array([[1.0, 0.0]]))
        b = tensor(np.array([[-1.0, 0.0]]))
        loss = cosine_similarity_loss(a, b)
        assert loss._data > 1.9

class TestL1Loss:

    def test_zero(self):
        pred = tensor(np.array([1.0, 2.0]))
        target = tensor(np.array([1.0, 2.0]))
        loss = l1_loss(pred, target)
        assert np.allclose(loss._data, 0.0)

    def test_backward(self):
        pred = tensor(np.array([2.0, 3.0]), requires_grad=True)
        target = tensor(np.array([1.0, 1.0]))
        loss = l1_loss(pred, target)
        loss.backward()
        assert pred._grad is not None

class TestSmoothL1:

    def test_alias(self):
        pred = tensor(np.array([1.5]))
        target = tensor(np.array([1.0]))
        loss = smooth_l1_loss(pred, target, beta=1.0)
        h = huber_loss(pred, target, delta=1.0)
        assert np.allclose(loss._data, h._data)

class TestNLLLoss:

    def test_basic(self):
        log_probs = tensor(np.array([[-0.1, -2.0, -5.0]]))
        targets = tensor(np.array([0]))
        loss = nll_loss(log_probs, targets)
        assert np.allclose(loss._data, 0.1)

    def test_backward(self):
        log_probs = tensor(np.array([[-1.0, -0.5]]), requires_grad=True)
        targets = tensor(np.array([0]))
        loss = nll_loss(log_probs, targets)
        loss.backward()
        assert log_probs._grad is not None

class TestAccuracy:

    def test_perfect(self):
        pred = tensor(np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]]))
        target = tensor(np.array([1, 0]))
        assert accuracy(pred, target) == 1.0

    def test_partial(self):
        pred = tensor(np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]]))
        target = tensor(np.array([0, 0]))
        assert accuracy(pred, target) == 0.5

class TestTopKAccuracy:

    def test_top2(self):
        pred = tensor(np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]))
        target = tensor(np.array([2, 0]))
        assert top_k_accuracy(pred, target, k=2) == 1.0

class TestPerplexity:

    def test_basic(self):
        assert perplexity(0.0) == 1.0
        assert perplexity(1.0) == pytest.approx(np.e, rel=0.0001)

class TestConfusionMatrix:

    def test_3class(self):
        pred = tensor(np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]))
        target = tensor(np.array([0, 1, 2]))
        cm = confusion_matrix(pred, target, num_classes=3)
        assert cm.shape == (3, 3)
        assert np.all(np.diag(cm) == 1)

class TestPrecisionRecallF1:

    def test_perfect(self):
        pred = tensor(np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]))
        target = tensor(np.array([0, 1, 0]))
        p, r, f1 = precision_recall_f1(pred, target, average='macro')
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

class TestR2Score:

    def test_perfect(self):
        pred = tensor(np.array([1.0, 2.0, 3.0]))
        target = tensor(np.array([1.0, 2.0, 3.0]))
        assert r2_score(pred, target) == 1.0

    def test_poor(self):
        pred = tensor(np.array([3.0, 3.0, 3.0]))
        target = tensor(np.array([1.0, 2.0, 3.0]))
        assert r2_score(pred, target) < 1.0

class TestMAE:

    def test_zero(self):
        pred = tensor(np.array([1.0, 2.0]))
        target = tensor(np.array([1.0, 2.0]))
        assert mean_absolute_error(pred, target) == 0.0

    def test_nonzero(self):
        pred = tensor(np.array([2.0, 3.0]))
        target = tensor(np.array([1.0, 1.0]))
        assert mean_absolute_error(pred, target) == 1.5

class TestRMSE:

    def test_zero(self):
        pred = tensor(np.array([1.0, 2.0]))
        target = tensor(np.array([1.0, 2.0]))
        assert root_mean_squared_error(pred, target) == 0.0

    def test_nonzero(self):
        pred = tensor(np.array([3.0, 1.0]))
        target = tensor(np.array([1.0, 3.0]))
        assert root_mean_squared_error(pred, target) == pytest.approx(2.0)