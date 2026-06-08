"""the runtime modules that used to be reachable only from the e2e harness are
now exposed through triad.* and exercised here, so they are part of the live
surface rather than orphan code.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from runtime.compiler_runtime import _resolve_triad_stdlib

def _mod(name):
    m = _resolve_triad_stdlib(name)
    assert m is not None, f'{name} not resolvable'
    return m

def test_calibrate_reachable_and_runs():
    cal = _mod('triad.calibrate')
    from stdlib.regimes import resolve_regime
    p = resolve_regime('B0', N=32)
    result = cal.calibrate(p, ['crystallinity', 'dominant_k'], 4, False)
    assert 'final_observables' in result
    assert 'crystallinity' in result['final_observables']

def test_curriculum_reachable_and_runs():
    curr = _mod('triad.curriculum')
    from stdlib.regimes import resolve_regime
    p = resolve_regime('B0', N=32)
    sched = curr.SigmaSchedule(0.7, 1.8, 'linear', 3, 0.5)
    result = curr.run_curriculum(p, sched, 1.0, False)
    
    assert sched.sigma_at_step(sched.n_steps - 1) < 2.0
    assert result.final_observables['crystallinity'] >= 0.0

def test_memory_reachable_and_runs():
    mem = _mod('triad.memory')
    hm = mem.HopfieldMemory()
    
    assert hm.params.Lambda != 0 and hm.params.Gamma > 0 and hm.params.f_FDT > 0
    hm.store_pattern()
    r = hm.recall_with_seed(0, 4.0)
    assert 'best_overlap' in r

def test_sat_reachable_and_solves():
    sat = _mod('triad.sat')
    inst = sat.SATInstance(3, [[1, -2, 3], [-1, 2, 3], [1, 2, -3]])
    r = sat.solve_sat(inst, 'interference', 8.0)
    assert 'assignment' in r and 'violations' in r

def test_energy_reachable():
    energy = _mod('triad.energy')
    assert hasattr(energy, 'relax')
    assert hasattr(energy, 'EnergyReadout')

def test_qubits_reachable_and_bell_state():
    q = _mod('triad.qubits')
    circ = q.TriadCircuit(2)
    circ.h(0)
    circ.cnot(0, 1)
    result = circ.run(None, 400, 7)
    
    assert set(result.counts.keys()) <= {'00', '11'}
    assert sum(result.counts.values()) == 400

def test_qubits_substrate_gates_are_pillar_complete():
    from runtime.physics.qubits import substrate_gate, QubitState
    st = QubitState(1)
    
    for g in ('evolve', 'kerr', 'decohere', 'memory', 'full'):
        out = substrate_gate(st, gate_type=g, T=0.3, dt=0.01)
        assert out.n_qubits == 1

def test_consciousness_reachable_and_runs():
    cons = _mod('triad.consciousness')
    from runtime.core.solver import TriadParams, integrate
    p = TriadParams(N=32, L=16.0, T=2.0, dt=0.01, backend='cpu', record_every=2)
    out = integrate(p, auto_halve_dt=False)
    report = cons.report(out['density'], dx=out['dx'])
    for key in ('phi', 'lzc', 'causal_density', 'metastability'):
        assert key in report

if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  PASS {name}')
            except AssertionError as e:
                print(f'  FAIL {name}: {e}')
