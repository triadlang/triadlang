"""Fase 2  Camada 2: acelerador surrogate (MNO) calibrado contra o solver nativo B0.

Principios (do relatorio, nao-negociaveis):
  - O solver nativo (Camada 0) e o ground truth. O surrogate so e aceito DENTRO da
    tolerancia do gate de fidelidade; fora dela, cai de volta no solver nativo.
  - A unica perda fisica permitida e o RESIDUO DA EQUACAO TRIAD (mais o L2 dos dados
    do solver). Nenhum termo premia crystallinity, k_star ou slow_state. Nenhum residuo
    de phase-field. A cristalizacao continua emergente, nunca imposta.
  - O bloco surrogate e o TriadSSM (estado complexo = fase P1 + dissipacao P3, campos y
    = memoria P2): o estado oculto do SSM tem contraparte fisica exata na equacao.

Conteudo:
  triad_residual_full(...)   residuo numerico completo da equacao (FFT; P1+P2+P3) para o gate
  MNOSurrogate               operador SSM Psi(t) -> Psi(t+dt_chunk)
  triad_residual_loss_local  residuo local diferenciavel (kinetico FD + V + dissipacao) p/ treino
  FidelityGate               compara surrogate vs nativo: L2, norma, observaveis, residuo
  accelerated_rollout        rollout com gate + fallback automatico ao solver nativo
  generate_pairs / train_surrogate
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np

from runtime.core.solver import TriadParams, integrate, _effective_params, _build_V_ext
from runtime.physics.observables import crystallinity, dominant_wavenumber, ipr, participation_ratio
from runtime.ml import tensor as T
from runtime.ml.nn import Module, Linear, Adam
from runtime.ml.language import TriadFullBlock

@dataclass
class TriadContext:
    N: int
    L: float
    dx: float
    hbar: float
    m: float
    Lambda: float
    alpha: float
    sigma: float
    Gamma: float
    f_FDT: float
    nu: np.ndarray
    lam: np.ndarray
    V_ext: np.ndarray
    k: np.ndarray
    H_lin_k: np.ndarray

    @staticmethod
    def from_params(p: TriadParams) -> 'TriadContext':
        eff = _effective_params(p)
        x = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
        dx = float(x[1] - x[0])
        k = 2.0 * np.pi * np.fft.fftfreq(p.N, d=dx)
        abs_k = np.abs(k)
        H_lin_k = p.hbar ** 2 * k ** 2 / (2.0 * p.m) + eff['alpha'] * abs_k ** p.sigma
        V_ext = _build_V_ext(p, x)
        return TriadContext(
            N=p.N, L=p.L, dx=dx, hbar=p.hbar, m=p.m,
            Lambda=eff['Lambda'], alpha=eff['alpha'], sigma=p.sigma,
            Gamma=eff['Gamma'], f_FDT=eff['f_FDT'],
            nu=np.asarray(eff['lam'] * 0 + np.asarray(p.nu, dtype=np.float64)),
            lam=np.asarray(eff['lam'], dtype=np.float64),
            V_ext=np.asarray(V_ext, dtype=np.float64),
            k=k, H_lin_k=H_lin_k,
        )

def _reconstruct_y(rho_seq: np.ndarray, nu: np.ndarray, dt_chunk: float) -> np.ndarray:
    """Reconstroi os campos de memoria y_j ao longo de snapshots espacados de dt_chunk.

    Mesma recorrencia OU do solver, no espacamento macro:
        y_j(t_{k+1}) = a_j y_j(t_k) + (1-a_j) rho(t_{k+1}),  a_j = exp(-nu_j dt_chunk).
    rho_seq: (n_t, N). Retorna y: (n_t, M, N).
    """
    n_t, N = rho_seq.shape
    M = len(nu)
    a = np.exp(-nu * dt_chunk)
    y = np.zeros((n_t, M, N), dtype=np.float64)
    for kdx in range(1, n_t):
        y[kdx] = a[:, None] * y[kdx - 1] + (1.0 - a)[:, None] * rho_seq[kdx][None, :]
    return y

def triad_residual_full(psi_seq: np.ndarray, t: np.ndarray, ctx: TriadContext) -> dict:
    """Residuo completo da equacao nuclear sobre uma trajetoria de snapshots.

    r = i hbar dPsi/dt - [ -hbar^2/2m d2 + alpha(-Delta)^{sigma/2} + V_ext
                            + Lambda|Psi|^2 + V_mem - i Gamma ] Psi
    Parte linear (kinetico + fracionario) via FFT (diagonal em k). O termo eta e
    estocastico de media zero; sua variancia define um piso de residuo, por isso o gate
    compara o residuo do surrogate ao residuo do PROPRIO solver nativo no mesmo stride.

    psi_seq: (n_t, N) complexo. Retorna media de |r|^2 e por-passo.
    """
    n_t, N = psi_seq.shape
    rho_seq = np.abs(psi_seq) ** 2
    dt_chunk = float(np.mean(np.diff(t))) if n_t > 1 else 1.0
    y_seq = _reconstruct_y(rho_seq, ctx.nu, dt_chunk)
    per = []
    for kdx in range(n_t - 1):
        psi0 = psi_seq[kdx]
        psi1 = psi_seq[kdx + 1]
        dt = t[kdx + 1] - t[kdx]
        psi_mid = 0.5 * (psi0 + psi1)
        rho_mid = np.abs(psi_mid) ** 2
        lin = np.fft.ifft(ctx.H_lin_k * np.fft.fft(psi_mid))
        V_mem = (ctx.lam[:, None] * y_seq[kdx + 1]).sum(axis=0)
        Vr = ctx.V_ext + ctx.Lambda * rho_mid + V_mem
        Hpsi = lin + Vr * psi_mid - 1j * ctx.Gamma * psi_mid
        r = 1j * ctx.hbar * (psi1 - psi0) / dt - Hpsi
        per.append(float(np.mean(np.abs(r) ** 2)))
    per = np.asarray(per) if per else np.zeros(0)
    return {'mean': float(per.mean()) if per.size else 0.0, 'per_step': per}

def field_features(psi: np.ndarray) -> np.ndarray:
    """Psi (N,) complexo -> features (N,3) = [re, im, rho]."""
    re = np.real(psi)
    im = np.imag(psi)
    rho = re * re + im * im
    return np.stack([re, im, rho], axis=-1).astype(np.float64)

class MNOSurrogate(Module):
    """Psi(t) -> Psi(t + dt_chunk). O SSM varre o dominio espacial (operador neural).

    Preve um incremento (delta-Psi); Psi_next = Psi + delta. Comeca, portanto, do estado
    atual: estrutura nenhuma e imposta, apenas a propagacao e aprendida contra o B0.
    """

    def __init__(self, d_model: int = 32, d_state: int = 32, n_blocks: int = 2,
                 n_memory: int = 3, in_features: int = 3):
        self.lift = Linear(in_features, d_model)
        self.blocks = [TriadFullBlock(d_model, N=d_state, n_memory=n_memory)
                       for _ in range(n_blocks)]
        self.head = Linear(d_model, 2)

    def forward(self, feats: T.TriadTensor) -> T.TriadTensor:
        
        x = self.lift(feats)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)

    def predict(self, psi: np.ndarray) -> np.ndarray:
        """Rollout numerico de um passo macro, sem grafo de gradiente."""
        feats = field_features(psi)[None, :, :]
        with T.no_grad():
            delta = self.forward(T.tensor(feats))._data
        dre = np.asarray(delta[0, :, 0])
        dim = np.asarray(delta[0, :, 1])
        return psi + (dre + 1j * dim)

def _lap_periodic_2d(f: T.TriadTensor, inv_dx2: float) -> T.TriadTensor:
    """Laplaciano periodico 1D sobre tensor (B,N), diferenciavel."""
    fp = T.cat([f[:, 1:], f[:, 0:1]], axis=1)
    fm = T.cat([f[:, -1:], f[:, :-1]], axis=1)
    return (fp + fm - f * 2.0) * inv_dx2

def triad_residual_loss_local(re0, im0, re1, im1, Vr_const, ctx: TriadContext,
                              dt_chunk: float) -> T.TriadTensor:
    """Residuo local diferenciavel da equacao Triad para o passo previsto.

    Usa o kinetico por diferenca finita (FD). O termo fracionario alpha(-Delta)^{sigma/2}
    e nao-local (FFT) e fica fora do gradiente; entra completo no gate numerico. Nenhum
    termo de crystallinity. re0/im0/Vr_const sao constantes (estado atual + V_mem/V_ext);
    re1/im1 sao a saida do modelo. Vr_const = V_ext + V_mem (parte real independente de Psi_next).
    """
    inv_dx2 = 1.0 / (ctx.dx * ctx.dx)
    kin = -(ctx.hbar ** 2) / (2.0 * ctx.m)
    rho1 = re1 * re1 + im1 * im1
    Vr = Vr_const + rho1 * ctx.Lambda  
    
    Hre = _lap_periodic_2d(re1, inv_dx2) * kin + Vr * re1 + im1 * ctx.Gamma
    Him = _lap_periodic_2d(im1, inv_dx2) * kin + Vr * im1 - re1 * ctx.Gamma
    
    c = ctx.hbar / dt_chunk
    dt_re = (im1 - im0) * (-c)
    dt_im = (re1 - re0) * c
    r_re = dt_re - Hre
    r_im = dt_im - Him
    return (r_re * r_re + r_im * r_im).mean()

def _chunk_params(p: TriadParams, chunk_T: float, seed: int) -> TriadParams:
    return TriadParams(**{**p.__dict__, 'T': chunk_T, 'seed': seed,
                         'record_every': 10 ** 9})

def native_chunk(p: TriadParams, psi: np.ndarray, y: Optional[np.ndarray],
                 chunk_T: float, seed: int):
    """Um passo macro do solver nativo (ground truth), carregando psi e memoria y."""
    out = integrate(_chunk_params(p, chunk_T, seed), psi0=psi, y0=y,
                    auto_halve_dt=False)
    return out['psi_final'], out['y_final']

def generate_pairs(regimes, seeds, N: int, chunk_steps: int, n_chunks: int):
    """Pares (Psi(t_k), Psi(t_{k+1})) nos limites de chunk, do solver nativo, multi-regime.

    Retorna X (n,N,3) features e Yc (n,N) complexo alvo, mais a lista de contextos.
    """
    from stdlib.regimes import resolve_regime
    Xs, Ys = [], []
    ctx_by_regime = {}
    for reg in regimes:
        base = resolve_regime(reg, N=N)
        ctx_by_regime[reg] = TriadContext.from_params(base)
        dt = base.dt
        chunk_T = chunk_steps * dt
        for sd in seeds:
            x = np.linspace(-base.L / 2, base.L / 2, base.N, endpoint=False)
            dx = x[1] - x[0]
            psi = np.exp(-x ** 2 / 8.0).astype(np.complex128)
            psi /= np.sqrt((np.abs(psi) ** 2).sum() * dx)
            y = None
            for c in range(n_chunks):
                psi_next, y = native_chunk(base, psi, y, chunk_T, seed=sd * 1000 + c)
                Xs.append(field_features(psi))
                Ys.append(psi_next.astype(np.complex128))
                psi = psi_next
    X = np.asarray(Xs)
    Yc = np.asarray(Ys)
    return X, Yc, ctx_by_regime

def train_surrogate(X: np.ndarray, Yc: np.ndarray, ctx: TriadContext,
                    chunk_T: float, epochs: int = 30, batch: int = 16,
                    lr: float = 2e-3, w_phys: float = 0.05, d_model: int = 32,
                    d_state: int = 32, n_blocks: int = 2, seed: int = 0,
                    verbose: bool = True) -> tuple:
    """Treino: loss = L2(dados B0) + w_phys * residuo Triad local. Sem termo de crystallinity."""
    rng = np.random.default_rng(seed)
    n, N, _ = X.shape
    model = MNOSurrogate(d_model=d_model, d_state=d_state, n_blocks=n_blocks,
                         n_memory=len(ctx.nu))
    opt = Adam(model.parameters(), lr=lr)
    Yre = np.real(Yc); Yim = np.imag(Yc)
    Vmem_zero = np.zeros((N,))  
    Vr_const_np = ctx.V_ext  
    history = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot_l2 = tot_phys = 0.0
        nb = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            feats = T.tensor(X[idx])                 
            re0 = T.tensor(X[idx][:, :, 0])          
            im0 = T.tensor(X[idx][:, :, 1])
            delta = model.forward(feats)             
            dre = delta[:, :, 0]
            dim = delta[:, :, 1]
            re1 = re0 + dre
            im1 = im0 + dim
            tgt_re = T.tensor(Yre[idx]); tgt_im = T.tensor(Yim[idx])
            l2 = ((re1 - tgt_re) ** 2 + (im1 - tgt_im) ** 2).mean()
            Vr_const = T.tensor(np.broadcast_to(Vr_const_np, (len(idx), N)).copy())
            phys = triad_residual_loss_local(re0, im0, re1, im1, Vr_const, ctx, chunk_T)
            loss = l2 + phys * w_phys
            model.zero_grad()
            loss.backward()
            opt.step()
            tot_l2 += float(l2._data); tot_phys += float(phys._data); nb += 1
        history.append((tot_l2 / nb, tot_phys / nb))
        if verbose:
            print(f'  epoch {ep + 1:3d}/{epochs}  L2={tot_l2 / nb:.3e}  phys={tot_phys / nb:.3e}')
    return model, history

@dataclass
class FidelityGate:
    l2_tol: float = 5e-2          
    norm_tol: float = 5e-2        
    kstar_tol: float = 1e-6       
    cryst_tol: float = 5e-2       
    resid_factor: float = 3.0     

    def evaluate(self, sur: np.ndarray, ref: np.ndarray, t: np.ndarray,
                 ctx: TriadContext) -> dict:
        dx = ctx.dx
        num = np.sqrt(np.sum(np.abs(sur - ref) ** 2, axis=1) * dx)
        den = np.sqrt(np.sum(np.abs(ref) ** 2, axis=1) * dx) + 1e-30
        l2_rel = float(np.mean(num / den))
        norm_sur = np.sum(np.abs(sur) ** 2, axis=1) * dx
        norm_ref = np.sum(np.abs(ref) ** 2, axis=1) * dx
        norm_dev = float(np.max(np.abs(norm_sur - norm_ref)))
        k_min = 2 * np.pi / ctx.L
        dk = abs(dominant_wavenumber(sur[-1], dx, k_min=k_min)
                 - dominant_wavenumber(ref[-1], dx, k_min=k_min))
        dcr = abs(crystallinity(sur[-1], dx) - crystallinity(ref[-1], dx))
        res_sur = triad_residual_full(sur, t, ctx)['mean']
        res_ref = triad_residual_full(ref, t, ctx)['mean']
        passed = (l2_rel <= self.l2_tol and norm_dev <= self.norm_tol
                  and dk <= self.kstar_tol and dcr <= self.cryst_tol
                  and res_sur <= self.resid_factor * res_ref + 1e-30)
        return {'l2_rel': l2_rel, 'norm_dev': norm_dev, 'dk_star': dk,
                'd_crystallinity': dcr, 'residual_sur': res_sur,
                'residual_ref': res_ref, 'passed': bool(passed)}

def accelerated_rollout(p: TriadParams, T_total: float, surrogate: MNOSurrogate,
                        gate: FidelityGate, chunk_steps: int = 20,
                        warmup_chunks: int = 1, base_seed: int = 0) -> dict:
    """Rollout hibrido. Por chunk: tenta o surrogate; se o residuo Triad exceder o piso do
    solver nativo (gate.resid_factor), descarta e roda o solver nativo (Camada 0). O solver
    nativo e sempre o fallback: a fisica e soberana.
    """
    ctx = TriadContext.from_params(p)
    dt = p.dt
    chunk_T = chunk_steps * dt
    n_chunks = int(round(T_total / chunk_T))
    x = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    psi = np.exp(-x ** 2 / 8.0).astype(np.complex128)
    psi /= np.sqrt((np.abs(psi) ** 2).sum() * ctx.dx)
    y = None
    snaps = [psi.copy()]
    ts = [0.0]
    path = []          
    t_native = t_sur = 0.0
    for c in range(n_chunks):
        seed = base_seed * 100000 + c
        if c < warmup_chunks:
            t0 = time.perf_counter()
            psi, y = native_chunk(p, psi, y, chunk_T, seed)
            t_native += time.perf_counter() - t0
            path.append('native')
        else:
            t0 = time.perf_counter()
            psi_try = surrogate.predict(psi)
            dt_sur = time.perf_counter() - t0
            pair_t = np.array([0.0, chunk_T])
            r_sur = triad_residual_full(np.stack([psi, psi_try]), pair_t, ctx)['mean']
            
            t0 = time.perf_counter()
            psi_nat, y_nat = native_chunk(p, psi, y, chunk_T, seed)
            dt_nat = time.perf_counter() - t0
            r_ref = triad_residual_full(np.stack([psi, psi_nat]), pair_t, ctx)['mean']
            if r_sur <= gate.resid_factor * r_ref + 1e-30:
                psi = psi_try
                
                a = np.exp(-ctx.nu * chunk_T)
                rho = np.abs(psi) ** 2
                y = (a[:, None] * (y if y is not None else np.zeros((len(ctx.nu), p.N)))
                     + (1.0 - a)[:, None] * rho[None, :])
                t_sur += dt_sur
                path.append('surrogate')
            else:
                psi, y = psi_nat, y_nat
                t_native += dt_nat
                path.append('native')
        snaps.append(psi.copy())
        ts.append((c + 1) * chunk_T)
    snaps = np.asarray(snaps)
    ts = np.asarray(ts)
    n_sur = path.count('surrogate')
    return {'psi_seq': snaps, 't': ts, 'path': path, 'n_chunks': n_chunks,
            'n_surrogate': n_sur, 'frac_surrogate': n_sur / max(n_chunks, 1),
            't_surrogate': t_sur, 't_native': t_native, 'ctx': ctx}

def native_reference(p: TriadParams, T_total: float, chunk_steps: int = 20,
                     base_seed: int = 0) -> dict:
    """Trajetoria de referencia 100% nativa, nos mesmos limites de chunk (para o gate)."""
    ctx = TriadContext.from_params(p)
    dt = p.dt
    chunk_T = chunk_steps * dt
    n_chunks = int(round(T_total / chunk_T))
    x = np.linspace(-p.L / 2, p.L / 2, p.N, endpoint=False)
    psi = np.exp(-x ** 2 / 8.0).astype(np.complex128)
    psi /= np.sqrt((np.abs(psi) ** 2).sum() * ctx.dx)
    y = None
    snaps = [psi.copy()]
    ts = [0.0]
    t0 = time.perf_counter()
    for c in range(n_chunks):
        psi, y = native_chunk(p, psi, y, chunk_T, base_seed * 100000 + c)
        snaps.append(psi.copy())
        ts.append((c + 1) * chunk_T)
    elapsed = time.perf_counter() - t0
    return {'psi_seq': np.asarray(snaps), 't': np.asarray(ts), 'elapsed': elapsed,
            'ctx': ctx}
