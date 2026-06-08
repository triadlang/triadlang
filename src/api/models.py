from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

V_EXT_LITERAL = Optional[Literal['harmonic', 'double_well', 'gaussian_bump', 'ramp', 'lattice']]

class TriadParamsModel(BaseModel):
    L: float = 32.0
    N: int = 128
    dt: float = 0.005
    T: float = 20.0
    hbar: float = 1.0
    m: float = 1.0
    V_ext: V_EXT_LITERAL = 'harmonic'
    omega: float = 0.05
    Lambda: float = -0.5
    alpha: float = 0.15
    sigma: float = 1.5
    Gamma: float = 0.05
    f_FDT: float = 0.002
    nu: list[float] = Field(default=[2.0, 0.5, 0.1])
    lam: list[float] = Field(default=[-0.3, -0.2, -0.1])
    mode: Literal['full', 'linear', 'thermal'] = 'full'
    seed: int = 0
    record_every: int = 4
    D: int = 1
    backend: Literal['auto', 'numpy', 'cuda'] = 'auto'
    bc: Literal['periodic', 'absorbing'] = 'periodic'
    bc_width: float = 0.15
    step_mode: Literal['strang', 'exptrap'] = 'strang'
    trap_lambda: float = 0.5
    fdt_couple: bool = True
    kT: float = 1.0

    def to_triad_params(self):
        from runtime.core.solver import TriadParams
        d = self.model_dump()
        d['nu'] = tuple(d['nu'])
        d['lam'] = tuple(d['lam'])
        return TriadParams(**d)

    @classmethod
    def from_triad_params(cls, p) -> 'TriadParamsModel':
        v_ext = p.V_ext if isinstance(p.V_ext, str) or p.V_ext is None else None
        return cls(
            L=p.L, N=p.N, dt=p.dt, T=p.T, hbar=p.hbar, m=p.m,
            V_ext=v_ext, omega=p.omega, Lambda=p.Lambda,
            alpha=p.alpha, sigma=p.sigma, Gamma=p.Gamma, f_FDT=p.f_FDT,
            nu=list(p.nu), lam=list(p.lam), mode=p.mode, seed=p.seed,
            record_every=p.record_every, D=getattr(p, 'D', 1),
            backend=p.backend, bc=p.bc, bc_width=p.bc_width,
            step_mode=p.step_mode, trap_lambda=float(p.trap_lambda),
            fdt_couple=p.fdt_couple, kT=p.kT,
        )

class SolveRequest(BaseModel):
    params: TriadParamsModel = Field(default_factory=TriadParamsModel)
    regime: Optional[str] = None
    psi0_b64: Optional[str] = None
    stream: bool = False

class SolveResult(BaseModel):
    psi_final_b64: str
    x_b64: str
    dx: float
    t_final: float
    crystallinity: float
    k_star: float
    peak_density: float
    norm: float
    ipr: float
    fwhm: float
    params_used: TriadParamsModel

class BatchSolveRequest(BaseModel):
    params: TriadParamsModel = Field(default_factory=TriadParamsModel)
    regime: Optional[str] = None
    K: int = 8
    seeds: Optional[list[int]] = None

class BatchSolveResult(BaseModel):
    results: list[SolveResult]
    ensemble_crystallinity_mean: float
    ensemble_crystallinity_std: float

class CoupledRunRequest(BaseModel):
    regime: str = 'B0'
    n_substrates: int = 4
    coupling: Literal['ring', 'all2all', 'none'] = 'ring'
    kappa: float = -3.0
    T: float = 5.0
    N: int = 128
    seed: int = 0

class SubstrateResult(BaseModel):
    substrate_id: int
    name: str
    psi_final_b64: str
    crystallinity: float
    k_star: float
    peak_density: float
    norm: float
    ipr: float
    fwhm: float

class CoupledRunResult(BaseModel):
    substrates: list[SubstrateResult]
    elapsed: float

class ObservablesRequest(BaseModel):
    psi_b64: str
    dx: float
    k_cutoff: float = 1.0
    L: Optional[float] = None

class ObservablesResult(BaseModel):
    crystallinity: float
    k_star: float
    peak_density: float
    norm: float
    ipr: float
    fwhm: float
    participation_ratio: float

class PowerSpectrumRequest(BaseModel):
    psi_b64: str
    dx: float

class PowerSpectrumResult(BaseModel):
    k_b64: str
    P_b64: str

class LangRunRequest(BaseModel):
    source: str

class LangRunResult(BaseModel):
    stdout: str
    stderr: str
    ok: bool
    error: Optional[str] = None

class LangCheckResult(BaseModel):
    ok: bool
    errors: list[str]

class LangCompileResult(BaseModel):
    ir_json: str
    ok: bool
    error: Optional[str] = None

class LangFormatResult(BaseModel):
    source: str
    ok: bool
    error: Optional[str] = None

class RegimeInfo(BaseModel):
    name: str
    params: TriadParamsModel
