import pytest
import numpy as np
from runtime.tensor import tensor, TriadTensor, mse_loss, cross_entropy, no_grad
from runtime.nn import Module, Parameter, Linear, Conv1d, Conv2d, ReLU, Tanh, Sigmoid, Softmax, Sequential, Flatten, Dropout, BatchNorm1d, Embedding, LayerNorm, MultiHeadAttention, FeedForward, TransformerBlock, Transformer, SGD, Adam

class TestParameter:

    def test_from_data(self):
        p = Parameter(tensor([1.0, 2.0]))
        assert p.requires_grad is True
        assert np.allclose(p._data, [1.0, 2.0])

    def test_from_numpy(self):
        p = Parameter(np.array([3.0, 4.0]))
        assert p.requires_grad is True

class TestModule:

    def test_parameters_collection(self):
        l = Linear(4, 3)
        params = l.parameters()
        assert len(params) == 2
        assert all((isinstance(p, Parameter) for p in params))

    def test_zero_grad(self):
        l = Linear(4, 3)
        out = l(tensor(np.random.randn(2, 4)))
        out.backward(np.ones_like(out._data))
        l.zero_grad()
        for p in l.parameters():
            assert p._grad is None

    def test_sequential_parameters(self):
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        params = model.parameters()
        assert len(params) == 4

class TestLinear:

    def test_forward_shape(self):
        l = Linear(4, 3)
        x = tensor(np.random.randn(2, 4))
        out = l(x)
        assert out.shape == (2, 3)

    def test_forward_no_bias(self):
        l = Linear(4, 3, bias=False)
        assert l.bias is None
        x = tensor(np.random.randn(2, 4))
        out = l(x)
        assert out.shape == (2, 3)

    def test_backward(self):
        l = Linear(4, 3)
        x = tensor(np.random.randn(2, 4), requires_grad=True)
        out = l(x)
        out.backward(np.ones_like(out._data))
        assert x._grad is not None
        assert x._grad.shape == (2, 4)
        assert l.weight._grad is not None
        assert l.weight._grad.shape == (4, 3)

    def test_batched_3d(self):
        l = Linear(4, 3)
        x = tensor(np.random.randn(2, 5, 4))
        out = l(x)
        assert out.shape == (2, 5, 3)

class TestConv1d:

    def test_forward_shape(self):
        c = Conv1d(2, 4, kernel_size=3, padding=1)
        x = tensor(np.random.randn(2, 2, 8))
        out = c(x)
        assert out.shape == (2, 4, 8)

    def test_forward_no_batch(self):
        c = Conv1d(1, 2, kernel_size=3)
        x = tensor(np.random.randn(1, 10))
        out = c(x)
        assert out.shape == (2, 8)

    def test_backward(self):
        c = Conv1d(1, 2, kernel_size=3)
        x = tensor(np.random.randn(1, 10), requires_grad=True)
        out = c(x)
        out.backward(np.ones_like(out._data))
        assert x._grad is not None

class TestConv2d:

    def test_forward_shape(self):
        c = Conv2d(1, 4, kernel_size=3, padding=1)
        x = tensor(np.random.randn(2, 1, 8, 8))
        out = c(x)
        assert out.shape == (2, 4, 8, 8)

    def test_forward_no_batch(self):
        c = Conv2d(1, 2, kernel_size=3)
        x = tensor(np.random.randn(1, 8, 8))
        out = c(x)
        assert out.shape == (2, 6, 6)

    def test_backward(self):
        c = Conv2d(1, 2, kernel_size=3)
        x = tensor(np.random.randn(1, 6, 6), requires_grad=True)
        out = c(x)
        out.backward(np.ones_like(out._data))
        assert x._grad is not None

class TestActivations:

    def test_relu(self):
        r = ReLU()
        x = tensor([-2.0, 0.0, 3.0])
        out = r(x)
        assert np.allclose(out._data, [0.0, 0.0, 3.0])

    def test_tanh(self):
        t = Tanh()
        x = tensor([0.0])
        out = t(x)
        assert np.allclose(out._data, [0.0], atol=1e-07)

    def test_sigmoid(self):
        s = Sigmoid()
        x = tensor([0.0])
        out = s(x)
        assert np.allclose(out._data, [0.5], atol=1e-07)

    def test_softmax(self):
        s = Softmax()
        x = tensor([1.0, 2.0, 3.0])
        out = s(x)
        assert np.allclose(out._data.sum(), 1.0)

class TestSequential:

    def test_forward(self):
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        x = tensor(np.random.randn(3, 4))
        out = model(x)
        assert out.shape == (3, 2)

    def test_parameters(self):
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        params = model.parameters()
        assert len(params) == 4

class TestFlatten:

    def test_flatten(self):
        f = Flatten()
        x = tensor(np.random.randn(3, 4, 5))
        out = f(x)
        assert out.shape == (3, 20)

class TestDropout:

    def test_training_mode(self):
        d = Dropout(p=0.5)
        d.training = True
        x = tensor(np.ones(100))
        out = d(x)
        nonzero = out._data[out._data != 0]
        assert len(nonzero) < 100
        assert len(nonzero) > 0
        assert np.allclose(nonzero, 2.0)

    def test_eval_mode(self):
        d = Dropout(p=0.5)
        d.training = False
        x = tensor(np.ones(10))
        out = d(x)
        assert np.allclose(out._data, np.ones(10))

class TestBatchNorm1d:

    def test_training_forward(self):
        bn = BatchNorm1d(4)
        bn.training = True
        x = tensor(np.random.randn(8, 4))
        out = bn(x)
        assert out.shape == (8, 4)

    def test_eval_forward(self):
        bn = BatchNorm1d(4)
        bn.training = True
        x = tensor(np.random.randn(16, 4))
        _ = bn(x)
        bn.training = False
        x2 = tensor(np.random.randn(4, 4))
        out = bn(x2)
        assert out.shape == (4, 4)

class TestEmbedding:

    def test_forward(self):
        emb = Embedding(10, 4)
        idx = tensor([1, 3, 5])
        out = emb(idx)
        assert out.shape == (3, 4)

    def test_backward(self):
        emb = Embedding(10, 4)
        idx = tensor([1, 3])
        out = emb(idx)
        out.backward(np.ones_like(out._data))
        assert emb.weight._grad is not None

class TestLayerNorm:

    def test_forward(self):
        ln = LayerNorm(4)
        x = tensor(np.random.randn(2, 4))
        out = ln(x)
        assert out.shape == (2, 4)

class TestMultiHeadAttention:

    def test_forward(self):
        mha = MultiHeadAttention(d_model=16, n_heads=4, causal=True)
        x = tensor(np.random.randn(2, 8, 16))
        out = mha(x)
        assert out.shape == (2, 8, 16)

class TestTransformerBlock:

    def test_forward(self):
        block = TransformerBlock(d_model=16, n_heads=4, d_ff=64)
        x = tensor(np.random.randn(2, 8, 16))
        out = block(x)
        assert out.shape == (2, 8, 16)

class TestTransformer:

    def test_forward(self):
        model = Transformer(vocab_size=20, d_model=16, n_heads=4, n_layers=2, d_ff=64, max_len=32)
        idx = tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        out = model(idx)
        assert out.shape == (2, 3, 20)

    def test_parameters(self):
        model = Transformer(vocab_size=20, d_model=16, n_heads=4, n_layers=2, d_ff=64)
        params = model.parameters()
        assert len(params) > 0

    def test_backward(self):
        model = Transformer(vocab_size=10, d_model=8, n_heads=2, n_layers=1, d_ff=16, max_len=8)
        idx = tensor(np.array([[1, 2, 3]]))
        out = model(idx)
        loss = cross_entropy(out, tensor(np.array([2])))
        loss.backward()
        grads = [p._grad for p in model.parameters() if p._grad is not None]
        assert len(grads) > 0

class TestSGD:

    def test_step(self):
        l = Linear(4, 2)
        opt = SGD(l.parameters(), lr=0.1)
        x = tensor(np.random.randn(2, 4))
        out = l(x)
        out.backward(np.ones_like(out._data))
        w_before = l.weight._data.copy()
        opt.step()
        assert not np.allclose(l.weight._data, w_before)

    def test_momentum(self):
        l = Linear(4, 2)
        opt = SGD(l.parameters(), lr=0.01, momentum=0.9)
        x = tensor(np.random.randn(2, 4))
        out = l(x)
        out.backward(np.ones_like(out._data))
        opt.step()

    def test_zero_grad(self):
        l = Linear(4, 2)
        opt = SGD(l.parameters(), lr=0.01)
        x = tensor(np.random.randn(2, 4))
        out = l(x)
        out.backward(np.ones_like(out._data))
        opt.zero_grad()
        for p in l.parameters():
            assert p._grad is None

class TestAdam:

    def test_step(self):
        l = Linear(4, 2)
        opt = Adam(l.parameters(), lr=0.01)
        x = tensor(np.random.randn(2, 4))
        out = l(x)
        out.backward(np.ones_like(out._data))
        w_before = l.weight._data.copy()
        opt.step()
        assert not np.allclose(l.weight._data, w_before)

    def test_convergence(self):
        np.random.seed(42)
        l = Linear(2, 1, bias=False)
        opt = Adam(l.parameters(), lr=0.05)
        target_w = np.array([[1.5], [2.5]])
        for _ in range(200):
            x = tensor(np.random.randn(4, 2))
            y = tensor(x._data @ target_w)
            pred = l(x)
            loss = mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert np.allclose(l.weight._data, target_w, atol=0.3)

class TestTrainingLoop:

    def test_xor_training(self):
        np.random.seed(0)
        model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1), Sigmoid())
        opt = Adam(model.parameters(), lr=0.05)
        x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y_data = np.array([[0], [1], [1], [0]], dtype=np.float64)
        for epoch in range(500):
            x = tensor(x_data)
            y = tensor(y_data)
            pred = model(x)
            loss = mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with no_grad():
            pred = model(tensor(x_data))
            preds = (pred._data > 0.5).astype(int).flatten()
            expected = y_data.flatten().astype(int)
            assert np.allclose(preds, expected)

    def test_linear_regression(self):
        np.random.seed(42)
        model = Linear(3, 1, bias=True)
        opt = Adam(model.parameters(), lr=0.05)
        x_data = np.random.randn(32, 3)
        true_w = np.array([[1.0], [2.0], [-1.0]])
        y_data = x_data @ true_w + 0.5
        for _ in range(300):
            x = tensor(x_data)
            y = tensor(y_data)
            pred = model(x)
            loss = mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with no_grad():
            final_loss = mse_loss(model(tensor(x_data)), tensor(y_data))
            assert final_loss._data < 0.1