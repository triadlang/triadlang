from __future__ import annotations

from dataclasses import dataclass

from runtime.core.multi_runtime import MultiRuntime, Segment
from runtime.physics.observables_atoms import atom_centroids_nd, atom_count_nd
from stdlib.regimes import resolve_regime
from triad import ntri as np

_ASCII = ' .:-=+*#%@'

@dataclass
class Atom:
    x: float
    y: float
    z: float
    mass: float
    peak: float

    def pos(self):
        return (self.x, self.y, self.z)

    def __repr__(self):
        return (f'Atom(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, '
                f'mass={self.mass:.4f}, peak={self.peak:.4f})')

class Engine3D:

    def __init__(self, N: int = 32, L: float = 16.0, regime: str = 'B0',
                 dt: float = 0.005, seed: int = 0):
        self.N = int(N)
        self.L = float(L)
        params = resolve_regime(regime, seed=seed, L=self.L, N=self.N, dt=dt)
        params = type(params)(**{**params.__dict__, 'D': 3, 'bc': 'periodic'})
        self._rt = MultiRuntime(dt=dt, record_every=10 ** 9)
        rng = np.random.default_rng(seed)

        vac = 1e-3 * (rng.standard_normal((self.N,) * 3)
                      + 1j * rng.standard_normal((self.N,) * 3))
        self._sub = self._rt.add_substrate('world', params,
                                           psi=vac.astype(np.complex128))
        xp = self._sub.xp
        self.x = np.linspace(-self.L / 2, self.L / 2, self.N, endpoint=False)
        self.dx = float(self.x[1] - self.x[0])
        Xg, Yg, Zg = xp.meshgrid(xp.asarray(self.x), xp.asarray(self.x), xp.asarray(self.x), indexing='ij')
        self._X, self._Y, self._Z = Xg, Yg, Zg
        if self._sub.V_ext_static is None:
            self._sub.V_ext_static = xp.zeros((self.N,) * 3)
        else:
            v = self._sub.V_ext_static
            if hasattr(v, 'get') and type(v).__module__.startswith('cupy'):
                v = v.get()
            self._sub.V_ext_static = xp.asarray(np.array(v, dtype=np.float64))
            if self._sub.V_ext_static.shape != (self.N,) * 3:
                self._sub.V_ext_static = xp.zeros((self.N,) * 3)

        self._has_matter = False

    @property
    def t(self) -> float:
        return self._rt.global_t

    def step(self, dt_frame: float = 0.1):
        t_end = self._rt.global_t + float(dt_frame)
        seg = Segment(t_start=self._rt.global_t, t_end=t_end)
        self._rt.add_segment(seg)
        self._rt.run(verbose=False)
        self._rt.global_t = t_end
        self._rt.segments.clear()
        return self.t

    def spawn_atom(self, x: float, y: float, z: float,
                   width: float = 1.2, amp: float = 1.0,
                   vx: float = 0.0, vy: float = 0.0, vz: float = 0.0):
        xp = self._sub.xp
        r2 = ((self._X - x) ** 2 + (self._Y - y) ** 2 + (self._Z - z) ** 2)
        packet = amp * xp.exp(-r2 / (2.0 * width ** 2))
        phase = xp.exp(1j * (vx * self._X + vy * self._Y + vz * self._Z))
        self._sub.psi = self._sub.psi + packet * phase
        self._has_matter = True
        return self

    def place_shape(self, points, width: float = 1.0, amp: float = 1.0):
        for p in points:
            self.spawn_atom(float(p[0]), float(p[1]), float(p[2]),
                            width=width, amp=amp)
        return self

    def add_wall_box(self, center, size, height: float = 8.0, gap: float = 0.0):
        cx, cy, cz = (float(v) for v in center)
        sx, sy, sz = (float(v) for v in size)
        inside = ((np.abs(self._X - cx) <= sx / 2)
                  & (np.abs(self._Y - cy) <= sy / 2)
                  & (np.abs(self._Z - cz) <= sz / 2))
        if gap > 0.0:
            axes = [(sx, self._Y, self._Z, cy, cz),
                    (sy, self._X, self._Z, cx, cz),
                    (sz, self._X, self._Y, cx, cy)]
            axes.sort(key=lambda a: a[0])
            _, A, B, ca, cb = axes[0]
            hole = ((A - ca) ** 2 + (B - cb) ** 2) <= gap ** 2
            inside = inside & ~hole
        self._sub.V_ext_static = self._sub.V_ext_static + height * inside
        return self

    def add_well_sphere(self, center, radius: float, depth: float = 3.0):
        cx, cy, cz = (float(v) for v in center)
        r2 = ((self._X - cx) ** 2 + (self._Y - cy) ** 2 + (self._Z - cz) ** 2)
        self._sub.V_ext_static = self._sub.V_ext_static - depth * (r2 <= radius ** 2)
        return self

    def clear_potential(self):
        self._sub.V_ext_static = self._sub.xp.zeros((self.N,) * 3)
        return self

    def density(self) -> np.ndarray:
        from runtime.backend import asnumpy
        return asnumpy(np.abs(self._sub.psi) ** 2)

    def atoms(self, threshold_frac: float | None = None):
        if not self._has_matter:
            return []
        from runtime.backend import asnumpy
        psi = asnumpy(self._sub.psi)
        cents = atom_centroids_nd(psi, self.dx, threshold_frac=threshold_frac)
        rho = np.abs(psi) ** 2
        out = []
        for c in cents:
            pos = c['position']
            px = [float(v) for v in pos[:3]]
            r2 = ((self._X - px[0]) ** 2 + (self._Y - px[1]) ** 2
                  + (self._Z - px[2]) ** 2)
            ball = asnumpy(r2 <= (3.0 * self.dx) ** 2)
            mass = float(rho[ball].sum() * self.dx ** 3)
            peak = float(rho[ball].max()) if ball.any() else 0.0
            out.append(Atom(px[0], px[1], px[2], mass, peak))
        return out

    def atom_count(self, threshold_frac: float | None = None) -> int:
        if not self._has_matter:
            return 0
        from runtime.backend import asnumpy
        psi = asnumpy(self._sub.psi)
        return int(atom_count_nd(psi, self.dx, threshold_frac=threshold_frac, dim=3))

    def mass_in_sphere(self, center, radius: float) -> float:
        cx, cy, cz = (float(v) for v in center)
        r2 = ((self._X - cx) ** 2 + (self._Y - cy) ** 2 + (self._Z - cz) ** 2)
        rho = self.density()
        from runtime.backend import asnumpy
        return float(rho[asnumpy(r2 <= radius ** 2)].sum() * self.dx ** 3)

    def total_mass(self) -> float:
        return float(self.density().sum() * self.dx ** 3)

    def render_ascii(self, axis: str = 'z', chars: str = _ASCII) -> str:
        rho = self.density()
        ax = {'x': 0, 'y': 1, 'z': 2}[axis]
        img = rho.max(axis=ax)
        top = img.max()
        if top <= 0:
            top = 1.0
        idx = (img / top * (len(chars) - 1)).astype(int)
        lines = []
        for row in idx.T[::-1]:
            lines.append(''.join(chars[v] for v in row))
        return '\n'.join(lines)

def engine(N: int = 32, L: float = 16.0, regime: str = 'B0',
           dt: float = 0.005, seed: int = 0) -> Engine3D:
    return Engine3D(N=N, L=L, regime=regime, dt=dt, seed=seed)

class Memory3D:

    def __init__(self, side: int = 4, N: int = 32, L: float = 16.0,
                 seed: int = 0, settle_T: float = 0.3,
                 regime: str = 'anti_collapse'):
        self.side = int(side)
        self.dim = self.side * self.side
        self.eng = Engine3D(N=N, L=L, regime=regime, seed=seed)
        self.settle_T = float(settle_T)

        self.cell = L / self.side
        coords = (np.arange(self.side) + 0.5) * self.cell - L / 2
        self.radius = self.cell / 2
        self._sites = [(float(coords[i]), float(coords[j]), 0.0)
                       for i in range(self.side) for j in range(self.side)]

        self.depth = self._calibrate_depth()

    def _set_wells(self, depth: float):
        self.eng.clear_potential()
        for c in self._sites:
            self.eng.add_well_sphere(c, radius=self.radius, depth=depth)

    def _calibrate_depth(self, max_doublings: int = 8) -> float:
        prng = np.random.default_rng(0)
        proof = np.sign(prng.standard_normal(self.dim))
        proof[proof == 0] = 1.0
        psi0 = self.eng._sub.psi.copy()
        depth = 1.0
        for _ in range(max_doublings):
            self.eng._sub.psi = psi0.copy()
            self._set_wells(depth)
            self.store(proof)

            ok = True
            for _ in range(3):
                self.eng.step(self.settle_T)
                ok = ok and bool((self.read() == proof).all())
            self.eng._sub.psi = psi0.copy()
            if ok:
                return depth
            depth *= 2.0
        return depth

    def store(self, pattern) -> None:
        bits = np.sign(np.asarray(pattern, dtype=np.float64)).reshape(-1)
        if bits.size != self.dim:
            raise ValueError(f'pattern deve ter {self.dim} bits')
        for b, c in zip(bits, self._sites):
            if b > 0:
                self.eng.spawn_atom(c[0], c[1], c[2], width=self.radius, amp=1.0)
        self.eng.step(self.settle_T)

    def read(self) -> np.ndarray:
        self.eng.step(self.settle_T)
        masses = np.array([self.eng.mass_in_sphere(c, self.radius)
                           for c in self._sites])
        srt = np.sort(masses)
        gaps = np.diff(srt)
        cut = int(np.argmax(gaps))
        thr = float((srt[cut] + srt[cut + 1]) / 2.0)
        return np.where(masses > thr, 1.0, -1.0)

    def render(self) -> str:
        return self.eng.render_ascii(axis='z')

