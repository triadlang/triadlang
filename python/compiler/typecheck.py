from __future__ import annotations
from frontend.parser import Program, RegDecl, Op, LoopBlock, SegmentBlock, IfBlock, OutStmt, HaltStmt, Annotation, EvolveStmt, CoupleStmt, PairStmt, RingStmt, SequenceStmt, ObserveStmt, AssertStmt, SubstrateDecl, IdentRef
_KNOWN_METRICS = {'k_star', 'IPR', 'crystallinity', 'peak', 'FWHM', 'norm', 'stabilization', 'memory_persistence', 'bravais_family', 'cluster_count', 'cluster_centroids', 'slow_state', 'slow_state_late_mean', 'LZc', 'metastability', 'phi_id', 'pcist', 'causal_density', 'kuramoto', 'atom_count', 'atom_centroids', 'atom_separation', 'atom_persistence_late', 'atoms_per_region', 'atomicity_ratio'}
_KNOWN_PREDICATES = {'persistent', 'extended', 'structurally_open', 'non_trivial_memory', 'atomic', 'anti_collapsed'}
_KNOWN_PROJECTIONS = {'centroid_density', 'integrated_density', 'dominant_k_star', 'fourier_band', 'atoms_of_atoms', 'none'}

class TypeCheckError(Exception):

    def __init__(self, errors: list[str]):
        super().__init__('\n'.join(errors))
        self.errors = errors

def typecheck(prog: Program, regime_registry: set[str]) -> None:
    declared: set[str] = {'ZERO', 'ONE', 'TWO', 'NEG_ONE'}
    errors: list[str] = []

    def _check_known(name: str, line: int, what: str='substrate'):
        if name not in declared:
            errors.append(f"L{line}: unknown {what} {name!r} (declared: {sorted(declared)[:8]}{('...' if len(declared) > 8 else '')})")

    def walk_body(body):
        for stmt in body.body:
            if isinstance(stmt, Annotation):
                pass
            elif isinstance(stmt, RegDecl):
                if stmt.name in declared:
                    errors.append(f'L{stmt.line}: duplicate substrate {stmt.name!r}')
                else:
                    declared.add(stmt.name)
                if stmt.regime_name is not None and stmt.regime_name not in regime_registry:
                    errors.append(f'L{stmt.line}: unknown regime {stmt.regime_name!r} (available: {sorted(regime_registry)[:5]}...)')
            elif isinstance(stmt, EvolveStmt):
                _check_known(stmt.target.name, stmt.line)
            elif isinstance(stmt, CoupleStmt):
                _check_known(stmt.src.name, stmt.line, 'couple src')
                _check_known(stmt.dst.name, stmt.line, 'couple dst')
            elif isinstance(stmt, PairStmt):
                _check_known(stmt.a.name, stmt.line, 'pair a')
                _check_known(stmt.b.name, stmt.line, 'pair b')
            elif isinstance(stmt, RingStmt):
                for m in stmt.members:
                    _check_known(m.name, stmt.line, 'ring member')
            elif isinstance(stmt, SequenceStmt):
                _check_known(stmt.target.name, stmt.line, 'sequence target')
                for inp in stmt.inputs:
                    _check_known(inp.name, stmt.line, 'sequence input')
            elif isinstance(stmt, ObserveStmt):
                _check_known(stmt.target.name, stmt.line, 'OBSERVE')
                for m in stmt.metrics:
                    if m not in _KNOWN_METRICS:
                        errors.append(f'L{stmt.line}: unknown OBSERVE metric {m!r} (known: {sorted(_KNOWN_METRICS)[:5]}...)')
            elif isinstance(stmt, AssertStmt):
                _check_known(stmt.target.name, stmt.line, 'assert')
                if stmt.predicate not in _KNOWN_PREDICATES:
                    errors.append(f'L{stmt.line}: unknown assert predicate {stmt.predicate!r}')
            elif isinstance(stmt, SubstrateDecl):
                for m in stmt.composed_of:
                    _check_known(m.name, stmt.line, 'composed_of member')
                if stmt.name in declared:
                    errors.append(f'L{stmt.line}: duplicate substrate {stmt.name!r}')
                else:
                    declared.add(stmt.name)
                proj = stmt.properties.get('projection', 'none')
                if proj not in _KNOWN_PROJECTIONS:
                    errors.append(f'L{stmt.line}: unknown projection {proj!r} (known: {sorted(_KNOWN_PROJECTIONS)})')
                regime = stmt.properties.get('regime', None)
                if regime is not None and regime not in regime_registry:
                    errors.append(f'L{stmt.line}: unknown macro regime {regime!r}')
            elif isinstance(stmt, (Op, OutStmt, HaltStmt)):
                pass
            elif isinstance(stmt, LoopBlock):
                walk_body(stmt.body)
            elif isinstance(stmt, SegmentBlock):
                walk_body(stmt.body)
            elif isinstance(stmt, IfBlock):
                walk_body(stmt.then_body)
                if stmt.else_body is not None:
                    walk_body(stmt.else_body)
    walk_body(prog)
    if errors:
        raise TypeCheckError(errors)