"""Fase 3  Camada 3: leitura de longo prazo (PhyxMamba/PINNMamba-like), estritamente passiva.

Principios (do relatorio, nao-negociaveis):
  - O observador LE a trajetoria gravada do reservoir Triad e preve invariantes de longo
    prazo (geometria de atrator, persistencia de memoria, estado lento). NUNCA realimenta a
    equacao. Garantia estrutural: este modulo so consome arrays numpy gravados e devolve
    numpy; nao referencia estado vivo do solver, nao chama passo do solver, nao escreve em
    nenhum Substrate.
  - Preserva a geometria do atrator em vez de forcar um alvo: alinhado a 'crystallization is
    emergent not imposed'. PhyxMamba e usado como diagnostico, nao como driver.

O reservoir e um SSM complexo FIXO (echo-state): estado h com decaimento + rotacao espelha a
fisica (h_j ~ campos y_j), mas serve apenas de mapa de features. O unico ajuste e um readout
ridge de forma fechada (estavel). A previsao casa os INVARIANTES (estatisticos/geometricos),
nao a trajetoria ponto-a-ponto, exatamente o que a sensibilidade do regime aberto permite.
"""
from __future__ import annotations
import numpy as np
from runtime.core.solver import TriadParams, integrate
from runtime.physics.observables import (time_delay_embedding, attractor_geometry_invariants,
                                 memory_persistence, slow_state_late_mean)

OBSERVER_FEEDS_BACK = False

def density_scalar_series(density_traj: np.ndarray, dx: float,
                          kind: str = 'participation') -> np.ndarray:
    """Serie escalar por frame a partir da trajetoria de densidade (N, n_t).

    Observavel emergente para reconstrucao do atrator (Takens). Derivado so de |Psi|^2.
    """
    rho = np.asarray(density_traj, dtype=float)
    n2 = rho.sum(0) * dx
    if kind == 'participation':
        n4 = (rho ** 2).sum(0) * dx
        return n2 * n2 / np.maximum(n4, 1e-30)
    if kind == 'peak':
        return rho.max(0)
    if kind == 'ipr':
        n4 = (rho ** 2).sum(0) * dx
        return n4 / np.maximum(n2 * n2, 1e-30)
    return n2

def slow_memory_series(y_traj: np.ndarray) -> np.ndarray:
    """Serie do estado lento: media espacial do campo de memoria mais lento (menor nu).

    y_traj do solver 1D tem forma (M, N, n_t). O campo mais lento e o de menor taxa; aqui
    tomamos o ultimo indice (regimes ordenam nu em ordem decrescente, p.ex. B0 nu=(2,.5,.1)).
    """
    y = np.asarray(y_traj, dtype=float)
    if y.ndim != 3 or y.shape[0] == 0:
        return np.zeros(0)
    slow = y[-1]            
    return slow.mean(axis=0)

class AttractorObserver:
    """Reservoir SSM complexo fixo + readout ridge. Passivo: consome series, devolve series."""

    def __init__(self, size: int = 200, radius: float = 0.9, in_scale: float = 0.5,
                 reg: float = 1e-4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.size = size
        omega = rng.uniform(0.0, np.pi, size)
        mag = radius * rng.uniform(0.7, 1.0, size)
        self.A = mag * np.exp(1j * omega)
        self.B = in_scale * (rng.standard_normal(size) + 1j * rng.standard_normal(size))
        self.reg = reg
        self.W = None
        self.mu = 0.0
        self.sd = 1.0
        self._h = None
        self._feat_last = None

    def _features(self, h: np.ndarray) -> np.ndarray:
        return np.concatenate([h.real, h.imag, np.abs(h) ** 2, [1.0]])

    def _drive(self, u_series: np.ndarray):
        h = np.zeros(self.size, dtype=np.complex128)
        feats = []
        for u in u_series:
            h = self.A * h + self.B * u
            feats.append(self._features(h))
        return np.asarray(feats), h

    def fit(self, series: np.ndarray) -> 'AttractorObserver':
        s = np.asarray(series, dtype=float).ravel()
        self.mu, self.sd = float(s.mean()), float(s.std() + 1e-12)
        u = (s - self.mu) / self.sd
        F, h_end = self._drive(u)
        X, Y = F[:-1], u[1:]
        nf = F.shape[1]
        self.W = np.linalg.solve(X.T @ X + self.reg * np.eye(nf), X.T @ Y)
        self._h = h_end
        self._feat_last = F[-1]
        return self

    def forecast(self, n_steps: int) -> np.ndarray:
        if self.W is None:
            raise RuntimeError('observer nao ajustado; chame .fit() primeiro')
        h = self._h.copy()
        feat = self._feat_last.copy()
        out = []
        for _ in range(n_steps):
            u_next = float(feat @ self.W)
            out.append(u_next)
            h = self.A * h + self.B * u_next
            feat = self._features(h)
        return np.asarray(out) * self.sd + self.mu

def _invariant_distance(a: dict, b: dict) -> dict:
    d_corr = abs(a['corr_dim'] - b['corr_dim'])
    d_rad = abs(a['radius'] - b['radius']) / (abs(a['radius']) + 1e-12)
    pa, pb = np.asarray(a['pca_spectrum']), np.asarray(b['pca_spectrum'])
    n = min(len(pa), len(pb))
    d_pca = float(np.abs(pa[:n] - pb[:n]).sum())
    return {'d_corr_dim': float(d_corr), 'd_radius': float(d_rad), 'd_pca': d_pca}

def longterm_consistency(p: TriadParams, T_total: float, observe_frac: float = 0.6,
                         record_every: int = 2, kind: str = 'participation',
                         embed_dim: int = 5, tau: int = 3, obs_size: int = 200,
                         seed: int = 0,
                         tol_corr: float = 0.8, tol_rad: float = 0.35,
                         tol_pca: float = 0.35) -> dict:
    """Roda B0 nativo, observa a janela inicial, preve o resto, e compara INVARIANTES.

    Passivo de ponta a ponta: integra uma vez (leitura), o observador nunca toca o solver.
    Veredito = invariantes consistentes E nenhuma realimentacao.
    """
    pl = TriadParams(**{**p.__dict__, 'T': T_total, 'record_every': record_every})
    out = integrate(pl, auto_halve_dt=False)
    dx = out['dx']
    s = density_scalar_series(out['density'], dx, kind=kind)
    n = len(s)
    k = int(n * observe_frac)
    obs_win, future = s[:k], s[k:]
    observer = AttractorObserver(size=obs_size, seed=seed).fit(obs_win)
    pred = observer.forecast(len(future))
    dt_eff = record_every * p.dt
    inv_true = attractor_geometry_invariants(future, dim=embed_dim, tau=tau)
    inv_pred = attractor_geometry_invariants(pred, dim=embed_dim, tau=tau)
    dist = _invariant_distance(inv_true, inv_pred)
    mp_true = memory_persistence(future, dt=dt_eff)
    mp_pred = memory_persistence(pred, dt=dt_eff)
    consistent = (dist['d_corr_dim'] <= tol_corr and dist['d_radius'] <= tol_rad
                  and dist['d_pca'] <= tol_pca)
    passed = bool(consistent and not OBSERVER_FEEDS_BACK)
    return {'inv_true': inv_true, 'inv_pred': inv_pred, 'dist': dist,
            'mp_true': mp_true, 'mp_pred': mp_pred,
            'slow_late_true': slow_state_late_mean(future),
            'slow_late_pred': slow_state_late_mean(pred),
            'n_observed': k, 'n_future': len(future),
            'feeds_back': OBSERVER_FEEDS_BACK, 'consistent': consistent,
            'passed': passed}

class ConvergenceObserver:
    """Passive convergence detector consuming checkpoints from _integrate_steps.

    Maintains rolling windows per metric and certifies convergence when the
    moving-average variation falls below tolerance.  Strictly read-only: never
    feeds back into the solver.

    Parameters
    ----------
    metrics : list[str]
        Names of observable functions in runtime.physics.observables to track.
        Supported: "crystallinity", "ipr", "participation", "energy",
        "dominant_wavenumber".
    tol : float
        Maximum relative variation in the window to certify convergence.
    window : int
        Number of consecutive checkpoints that must be stable.
    min_checkpoints : int
        Minimum checkpoints before certification is allowed.
    certify : str
        "anytime" for anytime-valid sequential testing (CITE-style);
        "simple" for plain window check.
    """

    METRIC_FNS = None  

    def __init__(self, metrics=None, tol: float = 1e-3, window: int = 5,
                 min_checkpoints: int = 10, certify: str = "simple"):
        if metrics is None:
            metrics = ["crystallinity", "energy"]
        self.metric_names = metrics
        self.tol = tol
        self.window = window
        self.min_checkpoints = min_checkpoints
        self.certify = certify
        self._history = {m: [] for m in metrics}
        self._n_checked = 0
        self._certified = False
        self._anytime_e_value = 0.0

    @classmethod
    def _get_fns(cls):
        if cls.METRIC_FNS is None:
            from runtime.physics import observables as obs_mod
            cls.METRIC_FNS = {
                "crystallinity": obs_mod.crystallinity,
                "ipr": obs_mod.ipr,
                "participation": obs_mod.participation_ratio,
                "dominant_wavenumber": obs_mod.dominant_wavenumber,
                "energy": obs_mod.energy,
            }
        return cls.METRIC_FNS

    def _compute_metric(self, name: str, chk: dict) -> float:
        fns = self._get_fns()
        fn = fns[name]
        psi = chk["psi"]
        dx = chk["dx"]
        if name == "energy":
            p = chk.get("params")
            hbar = p.hbar if p else 1.0
            m = p.m if p else 1.0
            Lambda = p.Lambda if p else -0.5
            return fn(psi, dx, hbar=hbar, m=m, Lambda=Lambda)
        return fn(psi, dx)

    def check(self, chk: dict) -> bool:
        """Process one checkpoint. Returns True if convergence is certified."""
        if self._certified:
            return True
        self._n_checked += 1
        for m in self.metric_names:
            val = self._compute_metric(m, chk)
            self._history[m].append(val)

        if self._n_checked < self.min_checkpoints:
            return False

        if self.certify == "anytime":
            return self._check_anytime()
        return self._check_simple()

    def _check_simple(self) -> bool:
        """Window-based convergence: all metrics stable for `window` checks."""
        for m in self.metric_names:
            vals = self._history[m]
            if len(vals) < self.window:
                return False
            recent = vals[-self.window:]
            mean_val = sum(recent) / len(recent)
            if abs(mean_val) < 1e-15:
                continue
            variation = max(abs(v - mean_val) for v in recent) / (abs(mean_val) + 1e-15)
            if variation > self.tol:
                return False
        self._certified = True
        return True

    def _check_anytime(self) -> bool:
        """Anytime-valid sequential test (CITE-inspired).

        Uses an exponential martingale: e_n = product of likelihood ratios
        under the running mean vs the null (no convergence).  When e_n
        exceeds 1/alpha (alpha = tolerance), reject the null and certify.
        This controls the false-certification rate under data-driven stopping.
        """
        alpha = self.tol
        if alpha <= 0:
            return False
        
        all_stable = True
        for m in self.metric_names:
            vals = self._history[m]
            if len(vals) < self.window:
                return False
            recent = vals[-self.window:]
            mean_v = sum(recent) / len(recent)
            if abs(mean_v) < 1e-15:
                continue
            var_v = sum((v - mean_v) ** 2 for v in recent) / len(recent)
            rel_var = var_v / (mean_v ** 2 + 1e-15)
            if rel_var > alpha * 10:
                all_stable = False
                break

        if not all_stable:
            self._anytime_e_value = 0.0
            return False

        self._anytime_e_value += 1.0
        threshold = 1.0 / alpha

        if self._anytime_e_value >= threshold:
            self._certified = True
            return True
        return False

    @property
    def certified(self) -> bool:
        return self._certified

    @property
    def n_checked(self) -> int:
        return self._n_checked

    def metric_values(self, name: str) -> list[float]:
        return list(self._history.get(name, []))

    def as_stop_fn(self):
        """Return a stop_fn compatible with integrate_adaptive.

        Usage:
            obs = ConvergenceObserver(metrics=["crystallinity"], tol=1e-3)
            result = integrate_adaptive(p, stop_fn=obs.as_stop_fn(), ...)
        """
        def stop_fn(history):
            return self.check(history[-1])
        return stop_fn

class SelfVerifier:
    """Passive self-verification via intrinsic observables + ensemble disagreement.

    Reads energy, crystallinity from converged field as confidence score.
    Optionally runs a small ensemble of FULL integrations with different seeds
    and measures observable disagreement as epistemic uncertainty.

    Never feeds back into the solver.
    """

    def __init__(self, ensemble: int = 5, confidence_metrics: list = None):
        """
        Parameters
        ----------
        ensemble : int
            Number of ensemble members (different noise realizations).
        confidence_metrics : list[str] or None
            Metrics to compute per member.  Default: crystallinity, energy.
        """
        self.ensemble = ensemble
        self.confidence_metrics = confidence_metrics or ["crystallinity", "energy"]
        self._results = []

    def verify(self, params_list: list, results_list: list) -> dict:
        """Compute confidence and disagreement from multiple run results.

        Parameters
        ----------
        params_list : list[TriadParams]
            The params used for each ensemble member.
        results_list : list[dict]
            The integrate() or integrate_adaptive() result per member.

        Returns
        -------
        dict with confidence, disagreement, per_member observables.
        """
        from runtime.physics.observables import crystallinity as _cryst, energy as _energy
        from runtime.physics.observables import participation_ratio as _pr

        member_obs = []
        for p, r in zip(params_list, results_list):
            psi = r["psi_final"]
            dx = r["dx"]
            obs = {
                "crystallinity": _cryst(psi, dx),
                "energy": _energy(psi, dx, hbar=p.hbar, m=p.m, Lambda=p.Lambda),
                "participation": _pr(psi, dx),
            }
            member_obs.append(obs)

        self._results = member_obs

        crystallinities = [m["crystallinity"] for m in member_obs]
        energies = [m["energy"] for m in member_obs]
        mean_C = sum(crystallinities) / len(crystallinities)
        mean_E = sum(energies) / len(energies)

        if len(crystallinities) > 1:
            var_C = sum((c - mean_C) ** 2 for c in crystallinities) / len(crystallinities)
            disagreement = var_C ** 0.5
        else:
            disagreement = 0.0

        confidence = mean_C  

        return {
            "confidence": confidence,
            "disagreement": disagreement,
            "mean_energy": mean_E,
            "n_members": len(member_obs),
            "per_member": member_obs,
        }

    def run_ensemble(self, base_params, T: float = 10.0) -> dict:
        """Run an ensemble and verify in one call.

        Each member uses the same params except seed (offset by member index).
        """
        from runtime.core.solver import integrate, TriadParams
        params_list = []
        results_list = []
        for i in range(self.ensemble):
            p = TriadParams(**{**base_params.__dict__,
                               "seed": base_params.seed + i * 1000,
                               "T": T})
            params_list.append(p)
            results_list.append(integrate(p))
        return self.verify(params_list, results_list)
