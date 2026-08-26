import importlib
import sys
import types


def test_backend_falls_back_when_cuda_probe_fails(monkeypatch):
    fake_cupy = types.ModuleType('cupy')

    class Runtime:
        @staticmethod
        def getDeviceCount():
            raise RuntimeError('no CUDA device')

    fake_cupy.cuda = types.SimpleNamespace(runtime=Runtime())
    monkeypatch.setitem(sys.modules, 'cupy', fake_cupy)
    monkeypatch.delenv('TRIADLANG_BACKEND', raising=False)
    sys.modules.pop('runtime.backend', None)

    backend = importlib.import_module('runtime.backend')

    assert backend.cuda_available() is False
    assert backend.BACKEND == 'numpy'


def test_cpu_backend_does_not_import_cupy(monkeypatch):
    monkeypatch.setenv('TRIADLANG_BACKEND', 'cpu')
    sys.modules.pop('runtime.backend', None)
    sys.modules.pop('cupy', None)
    sys.modules.pop('mlx', None)
    sys.modules.pop('mlx.core', None)

    backend = importlib.import_module('runtime.backend')

    assert backend.BACKEND == 'numpy'
    assert 'cupy' not in sys.modules
    assert 'mlx.core' not in sys.modules


def test_metal_backend_uses_mlx_when_available(monkeypatch):
    fake_mlx_pkg = types.ModuleType('mlx')
    fake_mx = types.ModuleType('mlx.core')
    fake_mx.gpu = object()
    fake_mx.float32 = 'float32'
    fake_mx.float64 = 'float64'
    fake_mx.complex64 = 'complex64'
    fake_mx.array = lambda data, dtype=None: data
    fake_mx.sum = lambda data: sum(data)
    fake_mx.eval = lambda *args: None
    fake_mx.set_default_device = lambda device: None
    fake_mx.random = types.SimpleNamespace(
        seed=lambda seed: None,
        normal=lambda shape: [0.0] * (shape if isinstance(shape, int) else shape[0]),
    )
    monkeypatch.setenv('TRIADLANG_BACKEND', 'metal')
    monkeypatch.setitem(sys.modules, 'mlx', fake_mlx_pkg)
    monkeypatch.setitem(sys.modules, 'mlx.core', fake_mx)
    sys.modules.pop('runtime.backend', None)

    backend = importlib.import_module('runtime.backend')

    assert backend.metal_available() is True
    assert backend.BACKEND == 'metal'
    assert backend.get_xp('metal') is backend.xp


def test_ane_is_reported_but_not_ndarray_backend(monkeypatch):
    monkeypatch.setenv('TRIADLANG_BACKEND', 'ane')
    monkeypatch.setitem(sys.modules, 'coremltools', types.ModuleType('coremltools'))
    sys.modules.pop('runtime.backend', None)

    backend = importlib.import_module('runtime.backend')

    assert backend.ane_available() is True
    try:
        backend.get_xp('ane')
    except RuntimeError as exc:
        assert 'Core ML inference target' in str(exc)
    else:
        raise AssertionError("backend='ane' must not pretend to be an ndarray backend")
