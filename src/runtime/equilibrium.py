
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time

from triad import ntri as np

try:
    import cupy as _cp
except (ImportError, ModuleNotFoundError):
    _cp = None

REGIONS = ('cpu', 'gpu', 'disk')
MEM_REGIONS = ('vram', 'unified', 'ram', 'disk')

_STATIC_KEYS = ('cpu_gflops', 'gpu_gflops', 'h2d_gbps', 'cpu_membw_gbps',
                'disk_gbps', 'cpu_dispatch_us', 'gpu_launch_us',
                'vram_bw_gbps')
_GPU_KEYS = ('gpu_available', 'gpu_gflops', 'h2d_gbps', 'gpu_launch_us',
             'vram_bw_gbps', 'vram_free_gb')

_SAFE_DEFAULTS = {
    'gpu_available': False, 'gpu_gflops': 0.0, 'h2d_gbps': 0.0,
    'gpu_launch_us': float('inf'), 'vram_bw_gbps': 0.0, 'vram_free_gb': 0.0,
    'cpu_gflops': 10.0, 'cpu_membw_gbps': 5.0, 'disk_gbps': 0.5,
    'cpu_dispatch_us': 5.0, 'ram_free_gb': 1.0,
}

def hardware_fingerprint() -> str:

    parts = []
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    parts.append(line.split(':', 1)[1].strip())
                    break
    except OSError:
        parts.append('cpu?')
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    parts.append(line.split()[1])
                    break
    except OSError:
        parts.append('ram?')
    if _cp is not None:
        try:
            name = _cp.cuda.runtime.getDeviceProperties(0)['name']
            parts.append(name.decode() if isinstance(name, bytes) else str(name))
        except (RuntimeError, AttributeError, OSError):
            parts.append('no-gpu')
    else:
        parts.append('no-cupy')
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:16]

def _default_cache_path() -> str:
    env = os.environ.get('TRIAD_TELEMETRY_CACHE')
    if env:
        return env
    base = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
    return os.path.join(base, 'triadlang', 'telemetry.json')

def _validate_op(op):
    if not isinstance(op, dict):
        raise TypeError(f'op must be dict, got {type(op).__name__}')
    for k in ('flops', 'bytes', 'working_set'):
        v = float(op.get(k, 0.0))
        if not math.isfinite(v) or v < 0:
            raise ValueError(f'op[{k!r}] invalid: {op.get(k)!r}')
    if int(op.get('n_calls', 1)) < 1:
        raise ValueError(f'op["n_calls"] must be >= 1: {op.get("n_calls")!r}')
    da = op.get('data_at', 'cpu')
    if da not in REGIONS:
        raise ValueError(f'op["data_at"] must be one of {REGIONS}: {da!r}')

def _validate_buf(buf):
    if not isinstance(buf, dict):
        raise TypeError(f'buf must be dict, got {type(buf).__name__}')
    nbytes = float(buf.get('nbytes', -1))
    if not math.isfinite(nbytes) or nbytes <= 0:
        raise ValueError(f'buf["nbytes"] invalid: {buf.get("nbytes")!r}')
    if buf.get('consumer', 'gpu') not in ('cpu', 'gpu'):
        raise ValueError(f'buf["consumer"] must be cpu|gpu: {buf.get("consumer")!r}')
    if buf.get('access', 'seq') not in ('seq', 'random'):
        raise ValueError(f'buf["access"] must be seq|random: {buf.get("access")!r}')
    if int(buf.get('reuse', 1)) < 0:
        raise ValueError(f'buf["reuse"] must be >= 0: {buf.get("reuse")!r}')

class Telemetry:

    def __init__(self, ttl_s: float = 300.0, cache_path: str = None,
                 persist: bool = True, static_ttl_s: float = 7 * 86400.0):
        self.ttl = ttl_s
        self.static_ttl = static_ttl_s
        self._cache = {}
        self._lock = threading.RLock()
        self._gpu_block_until = 0.0
        self._persist = persist
        self._path = cache_path or _default_cache_path()
        self._fp = hardware_fingerprint()
        if persist:
            self._load_disk()

    def _load_disk(self):
        try:
            with open(self._path) as f:
                blob = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(blob, dict) or blob.get('fingerprint') != self._fp:
            return
        now_wall = time.time()
        now = time.monotonic()
        for key, entry in (blob.get('values') or {}).items():
            if key not in _STATIC_KEYS:
                continue
            try:
                val, ts = float(entry[0]), float(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
            if not math.isfinite(val) or val < 0:
                continue
            if now_wall - ts > self.static_ttl:
                continue

            self._cache[key] = (val, now - (now_wall - ts))

    def _save_disk(self):
        if not self._persist:
            return
        try:
            now_wall = time.time()
            now = time.monotonic()
            values = {}
            for key in _STATIC_KEYS:
                if key in self._cache:
                    val, ts = self._cache[key]
                    if math.isfinite(val):
                        values[key] = [val, now_wall - (now - ts)]
            os.makedirs(os.path.dirname(self._path) or '.', exist_ok=True)
            tmp = self._path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'version': 1, 'fingerprint': self._fp,
                           'values': values}, f)
            os.replace(tmp, self._path)
        except OSError:
            self._persist = False

    def _measure(self, key):
        if key == 'gpu_available':
            return _cp is not None and self._try_gpu()
        if key == 'cpu_gflops':
            a = np.random.standard_normal((512, 512)).astype(np.float32)
            t0 = time.perf_counter()
            for _ in range(3):
                a @ a
            dt = (time.perf_counter() - t0) / 3
            return 2 * 512**3 / dt / 1e9
        if key == 'gpu_gflops':
            if not self.get('gpu_available'):
                return 0.0
            a = _cp.random.standard_normal((1024, 1024), dtype=_cp.float32)
            (a @ a); _cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                a @ a
            _cp.cuda.Stream.null.synchronize()
            dt = (time.perf_counter() - t0) / 5
            return 2 * 1024**3 / dt / 1e9
        if key == 'h2d_gbps':
            if not self.get('gpu_available'):
                return 0.0
            a = np.random.standard_normal(8_000_000).astype(np.float32)
            _cp.asarray(a); _cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                _cp.asarray(a)
            _cp.cuda.Stream.null.synchronize()
            return a.nbytes * 3 / (time.perf_counter() - t0) / 1e9
        if key == 'cpu_membw_gbps':
            a = np.random.standard_normal(16_000_000)
            t0 = time.perf_counter()
            for _ in range(3):
                a * 1.0001
            return a.nbytes * 2 * 3 / (time.perf_counter() - t0) / 1e9
        if key == 'disk_gbps':

            return min(self.get('cpu_membw_gbps') / 4.0, 2.0)
        if key == 'cpu_dispatch_us':
            a = np.ones((8, 8), dtype=np.float32)
            t0 = time.perf_counter()
            for _ in range(2000):
                a @ a
            return (time.perf_counter() - t0) / 2000 * 1e6
        if key == 'vram_free_gb':
            if not self.get('gpu_available'):
                return 0.0
            free, total = _cp.cuda.runtime.memGetInfo()
            return free / 1e9
        if key == 'vram_bw_gbps':
            if not self.get('gpu_available'):
                return 0.0
            a = _cp.random.standard_normal(32_000_000, dtype=_cp.float32)
            (a * 2); _cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                a * 2
            _cp.cuda.Stream.null.synchronize()
            return a.nbytes * 2 * 5 / (time.perf_counter() - t0) / 1e9
        if key == 'ram_free_gb':
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable'):
                        return int(line.split()[1]) / 1e6
            return 4.0
        if key == 'gpu_launch_us':
            if not self.get('gpu_available'):
                return float('inf')
            a = _cp.ones(8, dtype=_cp.float32)
            (a * 2); _cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                a * 2
            _cp.cuda.Stream.null.synchronize()
            return (time.perf_counter() - t0) / 200 * 1e6
        raise KeyError(key)

    def _try_gpu(self):
        try:
            _cp.cuda.runtime.getDeviceCount()
            return True
        except (RuntimeError, AttributeError, OSError):
            return False

    def get(self, key):
        with self._lock:
            now = time.monotonic()
            ttl = self.static_ttl if key in _STATIC_KEYS else self.ttl
            if key in self._cache:
                val, ts = self._cache[key]
                if now - ts < ttl:
                    return val

            if key in _GPU_KEYS and now < self._gpu_block_until:
                return _SAFE_DEFAULTS[key]
            try:
                val = self._measure(key)
            except KeyError:
                raise
            except (ValueError, RuntimeError, TypeError, OSError):

                if key in _GPU_KEYS:
                    self._gpu_block_until = now + self.ttl
                val = _SAFE_DEFAULTS.get(key, 0.0)
            self._cache[key] = (val, now)
            if key in _STATIC_KEYS:
                self._save_disk()
            return val

    def update(self, key: str, value: float):

        if key not in _SAFE_DEFAULTS:
            raise KeyError(key)
        v = float(value)
        if not math.isfinite(v) or v < 0:
            return
        with self._lock:
            self._cache[key] = (v, time.monotonic())
            if key in _STATIC_KEYS:
                self._save_disk()

    def notify_failure(self, region: str = 'gpu'):

        if region not in REGIONS:
            raise ValueError(f'region must be one of {REGIONS}: {region!r}')
        with self._lock:
            if region == 'gpu':
                self._cache.pop('vram_free_gb', None)
                self._cache.pop('gpu_available', None)

    def estimate(self, op) -> dict:

        _validate_op(op)
        flops = op.get('flops', 0.0)
        nbytes = op.get('bytes', 0.0)
        data_at = op.get('data_at', 'cpu')
        n_calls = op.get('n_calls', 1)
        big_mem = op.get('working_set', nbytes)

        out = {}

        t_cpu = flops / (self.get('cpu_gflops') * 1e9 + 1) \
            + nbytes / (self.get('cpu_membw_gbps') * 1e9 + 1) \
            + n_calls * self.get('cpu_dispatch_us') * 1e-6
        if data_at == 'gpu':
            t_cpu += nbytes / (self.get('h2d_gbps') * 1e9 + 1)
        if data_at == 'disk':
            t_cpu += nbytes / (self.get('disk_gbps') * 1e9 + 1)

        ram_free = self.get('ram_free_gb') * 1e9
        if big_mem > ram_free * 0.9:
            t_cpu += (big_mem - ram_free * 0.9) / (self.get('disk_gbps') * 1e9 + 1) * 3
        out['cpu'] = t_cpu

        if self.get('gpu_available'):
            t_gpu = flops / (self.get('gpu_gflops') * 1e9 + 1) \
                + n_calls * self.get('gpu_launch_us') * 1e-6
            if data_at == 'disk':
                bw = min(self.get('h2d_gbps'), self.get('disk_gbps'))
                t_gpu += nbytes / (bw * 1e9 + 1)
            elif data_at != 'gpu':
                t_gpu += nbytes / (self.get('h2d_gbps') * 1e9 + 1)

            vram_free = self.get('vram_free_gb') * 1e9
            if big_mem > vram_free * 0.9:
                t_gpu += big_mem / (self.get('h2d_gbps') * 1e9 + 1) * 4
            out['gpu'] = t_gpu
        else:
            out['gpu'] = float('inf')

        t_disk = flops / (self.get('cpu_gflops') * 1e9 + 1) \
            + big_mem / (self.get('disk_gbps') * 1e9 + 1) \
            + n_calls * 20e-6
        if data_at != 'disk':
            t_disk += big_mem / (self.get('disk_gbps') * 1e9 + 1)
        out['disk'] = t_disk
        return out

def estimate_buffer(tel: Telemetry, buf: dict) -> dict:

    _validate_buf(buf)
    nbytes = buf['nbytes']
    consumer = buf.get('consumer', 'gpu')
    reuse = max(1, buf.get('reuse', 1))
    access = buf.get('access', 'seq')
    rand_pen = 4.0 if access == 'random' else 1.0

    vram_free = tel.get('vram_free_gb') * 1e9
    ram_free = tel.get('ram_free_gb') * 1e9
    vram_bw = tel.get('vram_bw_gbps') * 1e9 + 1
    h2d = tel.get('h2d_gbps') * 1e9 + 1
    membw = tel.get('cpu_membw_gbps') * 1e9 + 1
    disk = tel.get('disk_gbps') * 1e9 + 1

    out = {}

    if tel.get('gpu_available') and nbytes <= vram_free * 0.9:
        per_pass = nbytes / (vram_bw if consumer == 'gpu' else h2d)
        out['vram'] = nbytes / h2d + reuse * per_pass
    else:
        out['vram'] = float('inf')

    if tel.get('gpu_available') and nbytes <= (vram_free + ram_free) * 0.9:
        cabido = min(nbytes, vram_free * 0.9)
        exced = nbytes - cabido
        per_pass = cabido / (vram_bw / 1.1) + exced / h2d
        if consumer == 'cpu':
            per_pass = nbytes / membw
        out['unified'] = nbytes / h2d + reuse * per_pass
    else:
        out['unified'] = float('inf')

    if nbytes <= ram_free * 0.9:
        per_pass = nbytes / (h2d if consumer == 'gpu' else membw)
        out['ram'] = reuse * per_pass
    else:
        out['ram'] = float('inf')

    out['disk'] = reuse * rand_pen * nbytes / disk \
        + (nbytes / disk if buf.get('data_at') != 'disk' else 0.0)
    return out

class DecisionField:

    def __init__(self, telemetry: Telemetry = None, seed: int = 0,
                 Lambda: float = -0.3, Sigma_lambda: float = 0.45,
                 nu: tuple = (5.0, 0.2, 0.04), kappa: float = 0.05,
                 T_bath: float = 0.01, dt: float = 0.08, n_steps: int = 25):
        self.tel = telemetry or Telemetry()
        self.rng = np.random.default_rng(seed)
        self.Lambda = Lambda
        self.nu = np.asarray(nu)
        self.lam = -np.asarray([Sigma_lambda * 0.6, Sigma_lambda * 0.3, Sigma_lambda * 0.1])
        self.kappa = kappa
        self.T_bath = T_bath
        self.dt = dt
        self.n_steps = n_steps

        self._lock = threading.RLock()
        self.y = np.zeros((len(self.nu), 3))
        self.y_mem = np.zeros((len(self.nu), len(MEM_REGIONS)))
        self.history = []
        self.last_steps = 0

    def _relax(self, c: np.ndarray, y: np.ndarray):

        n = len(c)
        finite = np.isfinite(c)
        c_min = c[finite].min()

        gamma = np.where(finite, 0.5 * c / max(c_min, 1e-12), 50.0)

        psi = (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n))
        psi = psi / (np.sqrt((psi.real ** 2 + psi.imag ** 2).sum()) + 1e-300)
        y = y.copy()
        drive = 0.3
        noise_amp = np.sqrt(2 * 0.5 * self.T_bath * self.dt) / np.sqrt(2)
        noise = noise_amp * (self.rng.standard_normal((self.n_steps, n))
                             + 1j * self.rng.standard_normal((self.n_steps, n)))
        damp = np.exp(-gamma * self.dt)
        nudt = self.nu[:, None] * self.dt
        dominant_since = 0
        last_arg = -1

        max_steps = self.n_steps * 4
        extra = noise_amp * (self.rng.standard_normal((max_steps - self.n_steps, n))
                             + 1j * self.rng.standard_normal((max_steps - self.n_steps, n)))
        noise = np.concatenate([noise, extra], axis=0)
        steps_used = max_steps
        rho_acc = np.zeros(n)
        acc_count = 0
        for s in range(max_steps):
            rho = psi.real ** 2 + psi.imag ** 2
            y += nudt * (rho[None, :] - y)
            V = self.Lambda * rho + (self.lam[:, None] * y).sum(axis=0)
            lap = self.kappa * (np.roll(psi, 1) + np.roll(psi, -1) - 2 * psi)
            psi = psi * np.exp(-1j * V * self.dt) * damp \
                + lap * self.dt + drive * self.dt + noise[s]
            rho_now = psi.real ** 2 + psi.imag ** 2
            if s >= self.n_steps // 2:
                rho_acc += rho_now
                acc_count += 1

            arg = int(np.argmax(rho_now))
            if arg == last_arg and rho_now[arg] > 0.6 * rho_now.sum():
                dominant_since += 1
                if dominant_since >= 5:
                    steps_used = s + 1
                    break
            else:
                dominant_since = 0
                last_arg = arg
        self.last_steps = steps_used

        final = rho_acc / max(acc_count, 1) if acc_count else \
            (psi.real ** 2 + psi.imag ** 2)
        return int(np.argmax(final)), y

    def place(self, op: dict) -> str:

        costs = self.tel.estimate(op)
        with self._lock:
            c = np.array([costs[r] for r in REGIONS])
            idx, y = self._relax(c, self.y)
            self.y = y
            choice = REGIONS[idx]
            self.history.append((choice, dict(costs)))
        return choice

    def place_buffer(self, buf: dict) -> str:

        costs = estimate_buffer(self.tel, buf)
        with self._lock:
            c = np.array([costs[r] for r in MEM_REGIONS])
            idx, y = self._relax(c, self.y_mem)
            self.y_mem = y
            choice = MEM_REGIONS[idx]
            self.history.append((choice, dict(costs)))
        return choice

class Equilibrium:

    def __init__(self, seed: int = 0, telemetry: Telemetry = None):
        self.tel = telemetry or Telemetry()
        self.field = DecisionField(telemetry=self.tel, seed=seed)

    def place(self, op: dict) -> str:
        return self.field.place(op)

    def matmul_device(self, M: int, K: int, N: int, n_calls: int = 1,
                      data_at: str = 'cpu') -> str:
        op = {'flops': 2.0 * M * K * N * n_calls,
              'bytes': (M * K + K * N) * 4.0,
              'data_at': data_at, 'n_calls': n_calls}
        r = self.field.place(op)
        return 'cpu' if r == 'disk' else r

    def buffer_home(self, nbytes: float, consumer: str = 'gpu',
                    reuse: int = 1, access: str = 'seq') -> str:
        return self.field.place_buffer({'nbytes': nbytes, 'consumer': consumer,
                                        'reuse': reuse, 'access': access})

    def tensor(self, data, op: dict = None):

        import importlib
        _t = importlib.import_module('runtime.ml.tensor')
        dev = 'cpu'
        if op is not None:
            r = self.field.place(op)
            dev = 'cuda' if r == 'gpu' else 'cpu'
        return _t.tensor(data, device=dev)

    def observe(self, op: dict, region: str, t_real: float):

        r = 'gpu' if region in ('cuda', 'gpu') else region
        if r not in REGIONS:
            raise ValueError(f'region must be one of {REGIONS}: {region!r}')
        _validate_op(op)
        if not math.isfinite(t_real) or t_real <= 0:
            return
        flops = op.get('flops', 0.0)
        nbytes = op.get('bytes', 0.0)
        n_calls = op.get('n_calls', 1)
        data_at = op.get('data_at', 'cpu')
        tel = self.tel
        if r == 'gpu':
            overhead = n_calls * tel.get('gpu_launch_us') * 1e-6
            if data_at == 'disk':
                bw = min(tel.get('h2d_gbps'), tel.get('disk_gbps'))
                overhead += nbytes / (bw * 1e9 + 1)
            elif data_at != 'gpu':
                overhead += nbytes / (tel.get('h2d_gbps') * 1e9 + 1)
            compute = t_real - overhead
            if flops > 0 and compute > 0:
                tel.update('gpu_gflops', flops / compute / 1e9)
        elif r == 'cpu':
            overhead = nbytes / (tel.get('cpu_membw_gbps') * 1e9 + 1) \
                + n_calls * tel.get('cpu_dispatch_us') * 1e-6
            compute = t_real - overhead
            if flops > 0 and compute > 0:
                tel.update('cpu_gflops', flops / compute / 1e9)
        else:

            big = op.get('working_set', nbytes)
            rest = t_real - flops / (tel.get('cpu_gflops') * 1e9 + 1) \
                - n_calls * 20e-6
            if big > 0 and rest > 0:
                tel.update('disk_gbps', big / rest / 1e9)

    def notify_failure(self, region: str = 'gpu'):
        r = 'gpu' if region == 'cuda' else region
        self.tel.notify_failure(r)

_AUTO = None
_AUTO_LOCK = threading.Lock()

def current_auto():

    return _AUTO

def _set_auto(eq):
    global _AUTO
    with _AUTO_LOCK:
        _AUTO = eq
        try:
            import importlib
            _t = importlib.import_module('runtime.ml.tensor')
            _t._AUTO_EQ = eq
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug('equilibrium: could not set _AUTO_EQ on tensor module: %s', exc)

def auto_on(eq: Equilibrium = None, seed: int = 0) -> Equilibrium:

    e = eq if eq is not None else Equilibrium(seed=seed)
    _set_auto(e)
    return e

def auto_off():
    _set_auto(None)

class _AutoCtx:
    def __init__(self, eq):
        self.eq = eq
        self._prev = None

    def __enter__(self):
        self._prev = current_auto()
        _set_auto(self.eq)
        return self.eq

    def __exit__(self, *exc):
        _set_auto(self._prev)
        return False

def auto(eq: Equilibrium = None, seed: int = 0) -> _AutoCtx:

    return _AutoCtx(eq if eq is not None else Equilibrium(seed=seed))
