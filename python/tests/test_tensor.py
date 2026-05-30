import pytest
import numpy as np
from runtime.tensor import TriadTensor, tensor, zeros, ones, randn, rand, arange, linspace, eye, exp, log, sqrt, tanh, sigmoid, relu, softmax, cross_entropy, mse_loss, cat, stack, bmm, transpose, layer_norm, no_grad, _ensure_tensor

class TestTriadTensorCreation:

    def test_from_list(self):
        t = tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert np.allclose(t._data, [1.0, 2.0, 3.0])

    def test_from_int(self):
        t = tensor(5)
        assert t.shape == ()
        assert t._data == 5.0

    def test_from_numpy(self):
        a = np.array([1.0, 2.0])
        t = tensor(a)
        assert np.allclose(t._data, a)

    def test_requires_grad(self):
        t = tensor([1.0, 2.0], requires_grad=True)
        assert t.requires_grad is True
        t2 = tensor([1.0])
        assert t2.requires_grad is False

    def test_from_tensor(self):
        t1 = tensor([1.0, 2.0])
        t2 = tensor(t1)
        assert np.allclose(t2._data, t1._data)

class TestTriadTensorProperties:

    def test_shape(self):
        t = tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.shape == (2, 2)

    def test_ndim(self):
        t = tensor([1.0])
        assert t.ndim == 1
        t2 = tensor([[1.0]])
        assert t2.ndim == 2

    def test_size(self):
        t = tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.size == 4

    def test_T(self):
        t = tensor([[1.0, 2.0], [3.0, 4.0]])
        tt = t.T
        assert tt.shape == (2, 2)
        assert np.allclose(tt._data[0, 1], 3.0)

    def test_dtype(self):
        t = tensor([1.0])
        assert t.dtype == np.float64

class TestTriadTensorOps:

    def test_add(self):
        a = tensor([1.0, 2.0])
        b = tensor([3.0, 4.0])
        c = a + b
        assert np.allclose(c._data, [4.0, 6.0])

    def test_add_scalar(self):
        a = tensor([1.0, 2.0])
        c = a + 3.0
        assert np.allclose(c._data, [4.0, 5.0])

    def test_radd(self):
        a = tensor([1.0, 2.0])
        c = 3.0 + a
        assert np.allclose(c._data, [4.0, 5.0])

    def test_sub(self):
        a = tensor([5.0, 6.0])
        b = tensor([1.0, 2.0])
        c = a - b
        assert np.allclose(c._data, [4.0, 4.0])

    def test_mul(self):
        a = tensor([2.0, 3.0])
        b = tensor([4.0, 5.0])
        c = a * b
        assert np.allclose(c._data, [8.0, 15.0])

    def test_mul_scalar(self):
        a = tensor([2.0, 3.0])
        c = a * 2
        assert np.allclose(c._data, [4.0, 6.0])

    def test_rmul(self):
        a = tensor([2.0, 3.0])
        c = 3.0 * a
        assert np.allclose(c._data, [6.0, 9.0])

    def test_div(self):
        a = tensor([6.0, 8.0])
        b = tensor([2.0, 4.0])
        c = a / b
        assert np.allclose(c._data, [3.0, 2.0])

    def test_neg(self):
        a = tensor([1.0, -2.0])
        c = -a
        assert np.allclose(c._data, [-1.0, 2.0])

    def test_pow(self):
        a = tensor([2.0, 3.0])
        c = a ** 2
        assert np.allclose(c._data, [4.0, 9.0])

    def test_matmul_2d(self):
        a = tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(c._data, expected)

    def test_matmul_1d(self):
        a = tensor([1.0, 2.0, 3.0])
        b = tensor([4.0, 5.0, 6.0])
        c = a @ b
        assert np.allclose(c._data, 32.0)

class TestTriadTensorReductions:

    def test_sum(self):
        t = tensor([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(t.sum()._data, 10.0)

    def test_sum_axis(self):
        t = tensor([[1.0, 2.0], [3.0, 4.0]])
        s = t.sum(axis=0)
        assert np.allclose(s._data, [4.0, 6.0])

    def test_mean(self):
        t = tensor([2.0, 4.0, 6.0])
        assert np.allclose(t.mean()._data, 4.0)

    def test_mean_axis(self):
        t = tensor([[1.0, 3.0], [5.0, 7.0]])
        m = t.mean(axis=1)
        assert np.allclose(m._data, [2.0, 6.0])

    def test_max(self):
        t = tensor([1.0, 5.0, 3.0])
        assert np.allclose(t.max()._data, 5.0)

    def test_min(self):
        t = tensor([1.0, 5.0, 3.0])
        assert np.allclose(t.min()._data, 1.0)

class TestTriadTensorReshape:

    def test_reshape(self):
        t = tensor([1.0, 2.0, 3.0, 4.0])
        r = t.reshape(2, 2)
        assert r.shape == (2, 2)
        assert np.allclose(r._data[0], [1.0, 2.0])

    def test_flatten(self):
        t = tensor([[1.0, 2.0], [3.0, 4.0]])
        f = t.flatten()
        assert f.shape == (4,)

    def test_item(self):
        t = tensor(42.0)
        assert t.item() == 42.0

    def test_len(self):
        t = tensor([1.0, 2.0, 3.0])
        assert len(t) == 3

    def test_iter(self):
        t = tensor([1.0, 2.0, 3.0])
        vals = [float(x._data) for x in t]
        assert vals == [1.0, 2.0, 3.0]

class TestTriadTensorComparison:

    def test_eq(self):
        a = tensor([1.0, 2.0])
        b = tensor([1.0, 3.0])
        c = a == b
        assert c._data[0] == True
        assert c._data[1] == False

    def test_lt(self):
        a = tensor([1.0, 3.0])
        b = tensor([2.0, 2.0])
        c = a < b
        assert c._data[0] == True
        assert c._data[1] == False

    def test_gt(self):
        a = tensor([3.0, 1.0])
        b = tensor([2.0, 2.0])
        c = a > b
        assert c._data[0] == True
        assert c._data[1] == False

class TestTriadTensorIndexing:

    def test_getitem(self):
        t = tensor([10.0, 20.0, 30.0])
        assert np.allclose(t[1]._data, 20.0)

    def test_setitem(self):
        t = tensor([1.0, 2.0, 3.0])
        t[1] = 99.0
        assert np.allclose(t._data[1], 99.0)

class TestTriadTensorAutograd:

    def test_add_backward(self):
        a = tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = tensor([4.0, 5.0, 6.0], requires_grad=True)
        c = a + b
        c.backward(np.ones(3))
        assert np.allclose(a._grad, [1.0, 1.0, 1.0])
        assert np.allclose(b._grad, [1.0, 1.0, 1.0])

    def test_mul_backward(self):
        a = tensor([2.0, 3.0], requires_grad=True)
        b = tensor([4.0, 5.0], requires_grad=True)
        c = a * b
        c.backward(np.ones(2))
        assert np.allclose(a._grad, [4.0, 5.0])
        assert np.allclose(b._grad, [2.0, 3.0])

    def test_sub_backward(self):
        a = tensor([5.0, 6.0], requires_grad=True)
        b = tensor([1.0, 2.0], requires_grad=True)
        c = a - b
        c.backward(np.ones(2))
        assert np.allclose(a._grad, [1.0, 1.0])
        assert np.allclose(b._grad, [-1.0, -1.0])

    def test_div_backward(self):
        a = tensor([6.0, 8.0], requires_grad=True)
        b = tensor([2.0, 4.0], requires_grad=True)
        c = a / b
        c.backward(np.ones(2))
        assert np.allclose(a._grad, [0.5, 0.25])
        assert np.allclose(b._grad, [-1.5, -0.5])

    def test_pow_backward(self):
        a = tensor([2.0, 3.0], requires_grad=True)
        c = a ** 2
        c.backward(np.ones(2))
        assert np.allclose(a._grad, [4.0, 6.0])

    def test_neg_backward(self):
        a = tensor([1.0, 2.0], requires_grad=True)
        c = -a
        c.backward(np.ones(2))
        assert np.allclose(a._grad, [-1.0, -1.0])

    def test_matmul_backward(self):
        a = tensor([[1.0, 2.0]], requires_grad=True)
        b = tensor([[3.0], [4.0]], requires_grad=True)
        c = a @ b
        c.backward(np.ones((1, 1)))
        assert np.allclose(a._grad, [[3.0, 4.0]])
        assert np.allclose(b._grad, [[1.0], [2.0]])

    def test_sum_backward(self):
        a = tensor([1.0, 2.0, 3.0], requires_grad=True)
        s = a.sum()
        s.backward()
        assert np.allclose(a._grad, [1.0, 1.0, 1.0])

    def test_mean_backward(self):
        a = tensor([2.0, 4.0, 6.0], requires_grad=True)
        m = a.mean()
        m.backward()
        assert np.allclose(a._grad, [1.0 / 3, 1.0 / 3, 1.0 / 3])

    def test_reshape_backward(self):
        a = tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        r = a.reshape(2, 2)
        r.backward(np.ones((2, 2)))
        assert np.allclose(a._grad, [1.0, 1.0, 1.0, 1.0])

    def test_chained_backward(self):
        a = tensor([1.0, 2.0], requires_grad=True)
        b = a * 2
        c = b + a
        c.backward(np.ones(2))
        assert np.allclose(a._grad, [3.0, 3.0])

    def test_no_grad_when_not_required(self):
        a = tensor([1.0, 2.0])
        b = tensor([3.0, 4.0])
        c = a + b
        assert c.requires_grad is False

    def test_backward_no_grad_context(self):
        with no_grad():
            a = tensor([1.0, 2.0], requires_grad=True)
            b = a * 2
            assert b.requires_grad is False

class TestTriadTensorBroadcasting:

    def test_add_broadcast(self):
        a = tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tensor([10.0, 20.0])
        c = a + b
        expected = np.array([[11.0, 22.0], [13.0, 24.0]])
        assert np.allclose(c._data, expected)

    def test_add_broadcast_backward(self):
        a = tensor([[1.0, 2.0]], requires_grad=True)
        b = tensor([10.0, 20.0], requires_grad=True)
        c = a + b
        c.backward(np.ones((1, 2)))
        assert np.allclose(a._grad, [[1.0, 1.0]])
        assert np.allclose(b._grad, [1.0, 1.0])

class TestFunctionalAPI:

    def test_zeros(self):
        t = zeros(3, 4)
        assert t.shape == (3, 4)
        assert np.allclose(t._data, 0.0)

    def test_ones(self):
        t = ones(2, 3)
        assert t.shape == (2, 3)
        assert np.allclose(t._data, 1.0)

    def test_randn(self):
        t = randn(10)
        assert t.shape == (10,)

    def test_rand(self):
        t = rand(5)
        assert t.shape == (5,)
        assert np.all(t._data >= 0)
        assert np.all(t._data <= 1)

    def test_arange(self):
        t = arange(5)
        assert np.allclose(t._data, [0, 1, 2, 3, 4])

    def test_arange_start_stop(self):
        t = arange(2, 7)
        assert np.allclose(t._data, [2, 3, 4, 5, 6])

    def test_linspace(self):
        t = linspace(0, 1, 5)
        assert np.allclose(t._data, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_eye(self):
        t = eye(3)
        assert np.allclose(t._data, np.eye(3))

    def test_exp(self):
        t = tensor([0.0, 1.0])
        r = exp(t)
        assert np.allclose(r._data, [1.0, np.e])

    def test_exp_backward(self):
        t = tensor([1.0, 2.0], requires_grad=True)
        r = exp(t)
        r.backward(np.ones(2))
        assert np.allclose(t._grad, [np.e, np.e ** 2])

    def test_log(self):
        t = tensor([1.0, np.e])
        r = log(t)
        assert np.allclose(r._data, [0.0, 1.0])

    def test_log_backward(self):
        t = tensor([2.0, 4.0], requires_grad=True)
        r = log(t)
        r.backward(np.ones(2))
        assert np.allclose(t._grad, [0.5, 0.25])

    def test_sqrt(self):
        t = tensor([4.0, 9.0])
        r = sqrt(t)
        assert np.allclose(r._data, [2.0, 3.0])

    def test_tanh(self):
        t = tensor([0.0])
        r = tanh(t)
        assert np.allclose(r._data, [0.0], atol=1e-07)

    def test_sigmoid(self):
        t = tensor([0.0])
        r = sigmoid(t)
        assert np.allclose(r._data, [0.5], atol=1e-07)

    def test_relu(self):
        t = tensor([-2.0, 0.0, 3.0])
        r = relu(t)
        assert np.allclose(r._data, [0.0, 0.0, 3.0])

    def test_relu_backward(self):
        t = tensor([-1.0, 0.5, 2.0], requires_grad=True)
        r = relu(t)
        r.backward(np.ones(3))
        assert np.allclose(t._grad, [0.0, 1.0, 1.0])

    def test_softmax(self):
        t = tensor([1.0, 2.0, 3.0])
        s = softmax(t)
        assert np.allclose(s._data.sum(), 1.0)
        assert s._data[2] > s._data[1] > s._data[0]

    def test_softmax_backward(self):
        t = tensor([1.0, 2.0, 3.0], requires_grad=True)
        s = softmax(t)
        s.backward(np.ones(3))
        assert t._grad is not None

    def test_cross_entropy(self):
        logits = tensor([[1.0, 2.0, 3.0]])
        targets = tensor([2])
        loss = cross_entropy(logits, targets)
        assert loss._data > 0

    def test_cross_entropy_backward(self):
        logits = tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        targets = tensor([2])
        loss = cross_entropy(logits, targets)
        loss.backward()
        assert logits._grad is not None
        assert np.allclose(logits._grad.sum(), 0.0, atol=1e-10)

    def test_mse_loss(self):
        pred = tensor([1.0, 2.0, 3.0])
        target = tensor([1.0, 2.0, 3.0])
        loss = mse_loss(pred, target)
        assert np.allclose(loss._data, 0.0)

    def test_mse_loss_nonzero(self):
        pred = tensor([1.0, 2.0])
        target = tensor([3.0, 4.0])
        loss = mse_loss(pred, target)
        assert np.allclose(loss._data, 4.0)

class TestBatchedOps:

    def test_bmm(self):
        a = tensor(np.random.randn(2, 3, 4))
        b = tensor(np.random.randn(2, 4, 5))
        c = bmm(a, b)
        assert c.shape == (2, 3, 5)

    def test_bmm_backward(self):
        a = tensor(np.random.randn(2, 3, 4), requires_grad=True)
        b = tensor(np.random.randn(2, 4, 5), requires_grad=True)
        c = bmm(a, b)
        c.backward(np.ones((2, 3, 5)))
        assert a._grad is not None
        assert b._grad is not None

    def test_transpose(self):
        t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = transpose(t, 0, 1)
        assert r.shape == (3, 2)
        assert np.allclose(r._data[0], [1.0, 4.0])

    def test_layer_norm(self):
        t = tensor([[1.0, 2.0, 3.0]])
        g = tensor([1.0, 1.0, 1.0])
        b = tensor([0.0, 0.0, 0.0])
        r = layer_norm(t, g, b)
        assert r.shape == t.shape

    def test_cat(self):
        a = tensor([1.0, 2.0])
        b = tensor([3.0, 4.0])
        c = cat([a, b])
        assert np.allclose(c._data, [1.0, 2.0, 3.0, 4.0])

    def test_stack(self):
        a = tensor([1.0, 2.0])
        b = tensor([3.0, 4.0])
        c = stack([a, b])
        assert c.shape == (2, 2)

class TestTriadTensorMisc:

    def test_detach(self):
        t = tensor([1.0, 2.0], requires_grad=True)
        d = t.detach()
        assert d.requires_grad is False

    def test_numpy(self):
        t = tensor([1.0, 2.0])
        a = t.numpy()
        assert isinstance(a, np.ndarray)

    def test_repr(self):
        t = tensor([1.0])
        r = repr(t)
        assert 'TriadTensor' in r

    def test_str(self):
        t = tensor([1.0, 2.0])
        assert '1.' in str(t)

    def test_float_conversion(self):
        t = tensor(3.5)
        assert float(t) == 3.5

    def test_int_conversion(self):
        t = tensor(3.0)
        assert int(t) == 3

    def test_bool_single(self):
        t = tensor(1.0)
        assert bool(t) is True
        t2 = tensor(0.0)
        assert bool(t2) is False

    def test_no_grad_context(self):
        with no_grad():
            a = tensor([1.0], requires_grad=True)
            b = a * 3
            assert b.requires_grad is False
        a2 = tensor([1.0], requires_grad=True)
        b2 = a2 * 3
        assert b2.requires_grad is True

    def test_zero_grad(self):
        t = tensor([1.0, 2.0], requires_grad=True)
        c = t.sum()
        c.backward()
        assert t._grad is not None
        t.zero_grad()
        assert t._grad is None