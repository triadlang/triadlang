from __future__ import annotations

from runtime.ml.nn import Module, Parameter
from runtime.ml.tensor import TriadTensor
from triad import ntri as np


class TriadFlow:

    def __init__(self, params, lr: float = 0.01, regime: str = 'B0',
                 seed: int = 0, freeze: bool = True):
        from stdlib.regimes import resolve_regime
        p = resolve_regime(regime, seed=seed, L=32.0, N=128, dt=0.005)
        self.params = [q for q in params]
        self.lr = float(lr)

        self.nu = np.asarray(p.nu, dtype=np.float64)
        self.lam = np.asarray(p.lam, dtype=np.float64)
        self.alpha = float(p.alpha)
        self.Gamma = float(p.Gamma)
        self.kappa = float(p.f_FDT) / max(float(p.Gamma), 1e-12)
        self.freeze_on = bool(freeze)
        self._rng = np.random.default_rng(seed)
        self._mem = [np.zeros((len(self.nu),) + q._data.shape) for q in self.params]

        self._window = max(2, int(round(1.0 / abs(float(self.lam[-1])))))
        self._hist = [np.zeros((self._window,) + q._data.shape) for q in self.params]
        self._bath = [np.zeros((self._window,) + q._data.shape) for q in self.params]
        self._hist_i = 0
        self._steps = 0

        self._temp = [np.ones_like(q._data) for q in self.params]

    def zero_grad(self):
        for q in self.params:
            q._grad = None

    def _laplacian_1d(self, w: np.ndarray) -> np.ndarray:
        flat = w.reshape(-1)
        lap = np.roll(flat, 1) + np.roll(flat, -1) - 2.0 * flat
        return lap.reshape(w.shape)

    def step(self):
        self._steps += 1
        gs = [q._grad if q._grad is not None else np.zeros_like(q._data)
              for q in self.params]
        decay = float(np.exp(-abs(float(self.lam[-1]))))
        for i, g in enumerate(gs):
            self._temp[i] = decay * self._temp[i] + (1.0 - decay) * (g ** 2)

        for i, (q, g) in enumerate(zip(self.params, gs)):
            F = -g

            m = self._mem[i]
            for k in range(len(self.nu)):
                m[k] += self.nu[k] * F - abs(float(self.lam[k])) * m[k]
            M = m.sum(axis=0)

            Dterm = self._laplacian_1d(q._data)

            sigma = np.sqrt(2.0 * self.Gamma * self._temp[i] * self.lr)
            eta = self._rng.standard_normal(q._data.shape) * sigma
            delta = self.lr * (F + self.alpha * M + self.kappa * Dterm) + eta

            h = self._hist[i]
            h[self._hist_i % h.shape[0]] = delta
            self._bath[i][self._hist_i % h.shape[0]] = sigma ** 2
            if self.freeze_on:
                mask = self._crystal_mask(i, g)
                delta = delta * mask
            q._data = q._data + delta
        self._hist_i += 1

    def _crystal_mask(self, i: int, g: np.ndarray) -> np.ndarray:
        h = self._hist[i]
        if self._hist_i < h.shape[0]:
            return np.ones_like(g)

        net = np.abs(h.mean(axis=0))
        se_bath = np.sqrt(self._bath[i].mean(axis=0) / h.shape[0])
        frozen = net <= se_bath
        return np.where(frozen, 0.0, 1.0)

    def crystal_fraction(self) -> float:
        if self._hist_i < self._window:
            return 0.0
        total = frozen = 0
        for i, h in enumerate(self._hist):
            net = np.abs(h.mean(axis=0))
            se_bath = np.sqrt(self._bath[i].mean(axis=0) / h.shape[0])
            frozen += int((net <= se_bath).sum())
            total += net.size
        return frozen / max(total, 1)

class Wavetriad(Module):

    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(in_features)
        self.wr = Parameter(TriadTensor(rng.standard_normal((in_features, out_features)) * scale))
        self.wi = Parameter(TriadTensor(rng.standard_normal((in_features, out_features)) * scale))
        self._eps = 1e-12

    def forward(self, x: TriadTensor) -> TriadTensor:
        if not isinstance(x, TriadTensor):
            x = TriadTensor(np.asarray(x, dtype=np.float64))
        from runtime.ml.tensor import sqrt as _sqrt
        shape = x._data.shape
        flat = x.reshape(-1, shape[-1]) if len(shape) > 2 else x
        a = flat @ self.wr
        b = flat @ self.wi
        out = _sqrt(a * a + b * b + self._eps)
        if len(shape) > 2:
            out = out.reshape(*shape[:-1], self.wr._data.shape[1])
        return out

    def parameters(self):
        return [self.wr, self.wi]

    def phase(self) -> np.ndarray:
        return np.arctan2(self.wi._data, self.wr._data)

    def amplitude(self) -> np.ndarray:
        return np.sqrt(self.wr._data ** 2 + self.wi._data ** 2)

class AttractorMemory:

    def __init__(self, dim: int, cells_per_bit: int = 4, regime: str = 'B0',
                 seed: int = 0, settle_T: float = 0.5):
        from runtime.core.multi_runtime import MultiRuntime, Segment
        from stdlib.regimes import resolve_regime
        self.dim = int(dim)
        self.cpb = int(cells_per_bit)
        self.N = max(64, self.dim * self.cpb)
        self.settle_T = float(settle_T)
        self._Segment = Segment
        params = resolve_regime(regime, seed=seed, L=32.0, N=self.N, dt=0.005)
        self._rt = MultiRuntime(dt=params.dt, record_every=10 ** 9)
        rng = np.random.default_rng(seed)
        vac = 1e-3 * (rng.standard_normal(self.N) + 1j * rng.standard_normal(self.N))
        self._sub = self._rt.add_substrate('memory', params,
                                           psi=vac.astype(np.complex128))
        self._rng = rng

    def _evolve(self, T: float):
        seg = self._Segment(t_start=self._rt.global_t,
                            t_end=self._rt.global_t + float(T))
        self._rt.add_segment(seg)
        self._rt.run(verbose=False)
        self._rt.global_t = seg.t_end
        self._rt.segments.clear()

    def _imprint(self, bits: np.ndarray, amp: float):
        start = (self.N - self.dim * self.cpb) // 2
        add = np.zeros(self.N, dtype=np.complex128)
        for k, b in enumerate(bits):
            phase = 0.0 if b > 0 else np.pi
            sl = slice(start + k * self.cpb, start + (k + 1) * self.cpb)
            add[sl] = amp * np.exp(1j * phase)
        psi = self._sub.psi
        if type(psi).__module__.startswith('cupy'):
            import cupy as _cp
            add = _cp.asarray(add)
        self._sub.psi = psi + add

    def _read(self) -> np.ndarray:
        start = (self.N - self.dim * self.cpb) // 2
        out = np.zeros(self.dim)
        for k in range(self.dim):
            sl = slice(start + k * self.cpb, start + (k + 1) * self.cpb)
            out[k] = 1.0 if float(np.real(self._sub.psi[sl].mean())) >= 0 else -1.0
        return out

    STORE_AMP = 1.0

    def store(self, pattern) -> None:
        bits = np.sign(np.asarray(pattern, dtype=np.float64))
        self._imprint(bits, amp=self.STORE_AMP)
        self._evolve(self.settle_T)

    def recall(self, cue, known_frac: float | None = None) -> dict:
        cue = np.asarray(cue, dtype=np.float64)
        known = cue != 0

        self._imprint(np.where(known, np.sign(cue), 0.0), amp=self.STORE_AMP)
        self._evolve(self.settle_T)
        rec = self._read()
        return {'pattern': rec, 'known': known}

def attractor_recall_score(memory: AttractorMemory, pattern, mask_frac: float,
                           seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pattern = np.sign(np.asarray(pattern, dtype=np.float64))
    cue = pattern.copy()
    hidden = rng.random(len(pattern)) < mask_frac
    cue[hidden] = 0.0
    out = memory.recall(cue)
    rec = out['pattern']
    if hidden.sum() == 0:
        return {'hidden': 0, 'accuracy_hidden': 1.0}
    acc = float((rec[hidden] == pattern[hidden]).mean())
    return {'hidden': int(hidden.sum()), 'accuracy_hidden': acc,
            'accuracy_total': float((rec == pattern).mean())}

class WaveFFN(Module):

    def __init__(self, d_model: int, d_ff: int, seed: int | None = None):
        from runtime.ml.nn import triad
        self.up = Wavetriad(d_model, d_ff, seed=seed)
        self.down = triad(d_ff, d_model)

    def forward(self, x: TriadTensor) -> TriadTensor:
        return self.down(self.up(x))

    def parameters(self):
        return self.up.parameters() + self.down.parameters()

class WaveTransformerBlock(Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 causal: bool = True, seed: int | None = None):
        from runtime.ml.nn import LayerNorm, MultiHeadAttention
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.ln1 = LayerNorm(d_model)
        self.ff = WaveFFN(d_model, d_ff, seed=seed)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x: TriadTensor) -> TriadTensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

    def parameters(self):
        out = []
        for m in (self.attn, self.ln1, self.ff, self.ln2):
            out.extend(m.parameters())
        return out

class FieldMemoryBank:

    def __init__(self, dim: int, slots: int = 4, spacing: int = 8,
                 regime: str = 'anti_collapse', seed: int = 0,
                 settle_T: float = 0.5):
        from runtime.core.multi_runtime import MultiRuntime, Segment
        from stdlib.regimes import resolve_regime
        self.dim = int(dim)
        self.slots = int(slots)
        self.sp = int(spacing)
        slot_w = self.dim * self.sp
        self.gap = self.sp * 2
        self.N = max(256, self.slots * (slot_w + self.gap))
        self.settle_T = float(settle_T)
        self._Segment = Segment
        params = resolve_regime(regime, seed=seed, L=float(self.N) / 4.0,
                                N=self.N, dt=0.005)
        self._rt = MultiRuntime(dt=params.dt, record_every=10 ** 9)
        rng = np.random.default_rng(seed)
        self._seed = seed
        vac = 1e-3 * (rng.standard_normal(self.N) + 1j * rng.standard_normal(self.N))
        self._sub = self._rt.add_substrate('bank', params,
                                           psi=vac.astype(np.complex128))
        self._used = 0

        self.depth = self._calibrate_depth()

    def _set_wells(self, depth: float):
        xp = self._sub.xp
        V = xp.zeros(self.N)
        idx = xp.arange(self.N)
        for k in range(self.slots):
            for j in range(self.dim):
                site = self._site(k, j)
                V -= depth * (xp.abs(idx - site) < self.sp / 2)
        self._sub.V_ext_static = V

    def _calibrate_depth(self, max_doublings: int = 8) -> float:

        prng = np.random.default_rng(self._seed)
        proof = np.sign(prng.standard_normal(self.dim))
        proof[proof == 0] = 1.0
        others = [np.sign(prng.standard_normal(self.dim)) for _ in range(self.slots - 1)]
        psi0 = self._sub.psi.copy()
        y0 = self._sub.y.copy()
        t0 = self._rt.global_t
        depth = 1.0
        for _ in range(max_doublings):
            self._sub.psi = psi0.copy()
            self._sub.y = y0.copy()
            self._rt.global_t = t0
            self._set_wells(depth)
            used_keep = self._used
            self._used = 0
            self.store(proof)

            ok = True
            stored = [proof]
            for other in others:
                self.store(other)
                stored.append(other)
                ok = ok and all(bool((self._read_slot(i) == pat).all())
                                for i, pat in enumerate(stored))
            for _ in range(self.slots):
                self._evolve(self.settle_T)
                ok = ok and all(bool((self._read_slot(i) == pat).all())
                                for i, pat in enumerate(stored))
            self._sub.psi = psi0.copy()
            self._sub.y = y0.copy()
            self._rt.global_t = t0
            self._used = used_keep
            if ok:
                return depth
            depth *= 2.0
        return depth

    def _site(self, k: int, j: int) -> int:
        slot_w = self.dim * self.sp
        return k * (slot_w + self.gap) + self.gap // 2 + j * self.sp

    def _evolve(self, T: float):
        seg = self._Segment(t_start=self._rt.global_t,
                            t_end=self._rt.global_t + float(T))
        self._rt.add_segment(seg)
        self._rt.run(verbose=False)
        self._rt.global_t = seg.t_end
        self._rt.segments.clear()

    def _read_slot(self, k: int) -> np.ndarray:
        _psi = self._sub.psi
        if type(_psi).__module__.startswith('cupy'):
            _psi = _psi.get()
        rho = np.abs(np.asarray(_psi)) ** 2
        half = self.sp // 2
        dens = np.array([rho[max(0, self._site(k, j) - half):self._site(k, j) + half].sum()
                         for j in range(self.dim)])

        srt = np.sort(dens)
        gaps = np.diff(srt)
        cut = int(np.argmax(gaps))
        thr = float((srt[cut] + srt[cut + 1]) / 2.0)
        return np.where(dens > thr, 1.0, -1.0)

    def store(self, pattern) -> int:
        bits = np.sign(np.asarray(pattern, dtype=np.float64))
        k = self._used % self.slots
        idx = np.arange(self.N)
        add = np.zeros(self.N, dtype=np.complex128)
        for j, b in enumerate(bits):
            if b > 0:
                site = self._site(k, j)
                add += 1.0 * (np.abs(idx - site) < self.sp / 2)
        psi = self._sub.psi
        if type(psi).__module__.startswith('cupy'):
            import cupy as _cp
            add = _cp.asarray(add)
        self._sub.psi = psi + add
        self._evolve(self.settle_T)
        self._used += 1
        return k

    def recall(self, cue) -> dict:
        cue = np.asarray(cue, dtype=np.float64)
        known = cue != 0
        self._evolve(self.settle_T)
        best_k, best_ov = -1, -2.0
        for k in range(min(self._used, self.slots)):
            r = self._read_slot(k)
            ov = float((r[known] == np.sign(cue[known])).mean()) if known.any() else 0.0
            if ov > best_ov:
                best_k, best_ov = k, ov
        rec = self._read_slot(best_k) if best_k >= 0 else np.zeros(self.dim)
        return {'pattern': rec, 'slot': best_k, 'cue_match': best_ov}
