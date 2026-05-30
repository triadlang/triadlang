from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
from runtime.solver import TriadParams
from runtime.equation_runtime import EquationRuntime, ReadoutConfig

@dataclass
class SubstrateConfig:
    name: str
    N: int = 128
    L: float = 32.0
    Lambda: float = -0.5
    alpha: float = 0.15
    sigma: float = 1.5
    Gamma: float = 0.05
    f_FDT: float = 0.002
    nu: tuple = (2.0, 0.5, 0.1)
    lam: tuple = (-0.3, -0.2, -0.1)
    V_ext: str = 'harmonic'
    omega: float = 0.05
    regime: str = 'B0'

@dataclass
class CouplingConfig:
    src: str
    dst: str
    kappa: float = -3.0
    mode: str = 'density'

@dataclass
class CompiledProgram:
    substrates: list[SubstrateConfig]
    couplings: list[CouplingConfig]
    readout: ReadoutConfig
    T: float = 5.0
    dt: float = 0.005
    backend: str = 'auto'
    metadata: dict = field(default_factory=dict)

class TriadCompiler:

    def compile(self, task: str, **kwargs) -> CompiledProgram:
        if task == 'classify':
            return self._compile_classify(**kwargs)
        elif task == 'generate':
            return self._compile_generate(**kwargs)
        elif task == 'remember':
            return self._compile_remember(**kwargs)
        elif task == 'couple':
            return self._compile_couple(**kwargs)
        elif task == 'sat':
            return self._compile_sat(**kwargs)
        else:
            raise ValueError(f'Unknown task: {task}')

    def _compile_classify(self, n_classes: int=2, n_features: int=128, depth: int=4, **kwargs) -> CompiledProgram:
        n_subs = max(depth, int(np.ceil(np.log2(n_classes + 1))))
        subs = []
        for i in range(n_subs):
            lam_scale = 1.0 / (i + 1)
            subs.append(SubstrateConfig(name=f'cls_{i}', N=n_features, Lambda=-0.5 * (1 + 0.2 * i), alpha=0.15, nu=(2.0, 0.5 / (i + 1), 0.1 / (i + 1)), lam=(-0.3 * lam_scale, -0.2 * lam_scale, -0.1 * lam_scale), omega=0.05 * (1 + 0.1 * i)))
        couplings = []
        for i in range(n_subs - 1):
            couplings.append(CouplingConfig(src=f'cls_{i}', dst=f'cls_{i + 1}', kappa=-3.0, mode='density'))
        couplings.append(CouplingConfig(src=f'cls_{n_subs - 1}', dst=f'cls_0', kappa=-1.5, mode='density'))
        readout = ReadoutConfig(what='full', fields=('crystallinity', 'k_star', 'peak', 'ipr', 'participation', 'fwhm', 'norm', 'density_pca'))
        return CompiledProgram(substrates=subs, couplings=couplings, readout=readout, metadata={'task': 'classify', 'n_classes': n_classes})

    def _compile_generate(self, pattern_type: str='crystal', N: int=128, **kwargs) -> CompiledProgram:
        if pattern_type == 'crystal':
            lam_split = (1.125, 0.375)
            nu_gen = (10.0, 0.5)
        elif pattern_type == 'filament':
            lam_split = (-0.3, -0.2, -0.1)
            nu_gen = (2.0, 0.5, 0.1)
        elif pattern_type == 'lattice':
            lam_split = (0.75, 0.25)
            nu_gen = (10.0, 0.5)
        else:
            lam_split = (-0.3, -0.2, -0.1)
            nu_gen = (2.0, 0.5, 0.1)
        subs = [SubstrateConfig(name='gen_0', N=N, Lambda=-8.0 if pattern_type == 'crystal' else -0.5, alpha=0.0 if pattern_type == 'crystal' else 0.15, nu=nu_gen, lam=lam_split, V_ext=None)]
        return CompiledProgram(substrates=subs, couplings=[], readout=ReadoutConfig(), T=15.0 if pattern_type == 'crystal' else 10.0, metadata={'task': 'generate', 'pattern_type': pattern_type})

    def _compile_remember(self, timescales: tuple=(1.0, 5.0, 20.0), N: int=128, **kwargs) -> CompiledProgram:
        nu = tuple((1.0 / ts for ts in timescales))
        lam = tuple((-0.3 / (i + 1) for i in range(len(timescales))))
        subs = [SubstrateConfig(name='mem_0', N=N, Lambda=-0.3, nu=nu, lam=lam, Gamma=0.02)]
        return CompiledProgram(substrates=subs, couplings=[], readout=ReadoutConfig(), metadata={'task': 'remember', 'timescales': timescales})

    def _compile_couple(self, n_substrates: int=3, topology: str='ring', kappa: float=-3.0, **kwargs) -> CompiledProgram:
        subs = [SubstrateConfig(name=f's{i}', regime='B0') for i in range(n_substrates)]
        couplings = []
        if topology == 'ring':
            for i in range(n_substrates):
                couplings.append(CouplingConfig(src=f's{i}', dst=f's{(i + 1) % n_substrates}', kappa=kappa))
        elif topology == 'full':
            for i in range(n_substrates):
                for j in range(n_substrates):
                    if i != j:
                        couplings.append(CouplingConfig(src=f's{i}', dst=f's{j}', kappa=kappa / n_substrates))
        elif topology == 'star':
            for i in range(1, n_substrates):
                couplings.append(CouplingConfig(src=f's{i}', dst=f's0', kappa=kappa))
                couplings.append(CouplingConfig(src=f's0', dst=f's{i}', kappa=kappa))
        return CompiledProgram(substrates=subs, couplings=couplings, readout=ReadoutConfig(), metadata={'task': 'couple', 'topology': topology})

    def _compile_sat(self, n_vars: int=3, n_clauses: int=9, **kwargs) -> CompiledProgram:
        N = max(64, n_vars * 8)
        return CompiledProgram(substrates=[SubstrateConfig(name='sat_0', N=N, Lambda=-5.0, alpha=0.0, V_ext=None, nu=(2.0, 0.5), lam=(-0.5, -0.2))], couplings=[], readout=ReadoutConfig(), T=10.0, metadata={'task': 'sat', 'n_vars': n_vars, 'n_clauses': n_clauses})

def instantiate(program: CompiledProgram, seed: int=0) -> EquationRuntime:
    n_subs = len(program.substrates)
    sub0 = program.substrates[0]
    rt = EquationRuntime(n_substrates=n_subs, regime=sub0.regime, N=sub0.N, coupling=_topology_from_couplings(program.couplings, n_subs), kappa=program.couplings[0].kappa if program.couplings else -3.0, seed=seed, backend=program.backend, readout=program.readout)
    rt._compiled_program = program
    return rt

def _topology_from_couplings(couplings: list[CouplingConfig], n_subs: int) -> str:
    if len(couplings) == n_subs:
        return 'ring'
    elif len(couplings) == n_subs * (n_subs - 1):
        return 'full'
    elif len(couplings) == 0:
        return 'ring'
    else:
        return 'ring'