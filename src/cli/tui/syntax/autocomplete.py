from __future__ import annotations

_KEYWORDS = [
    'let', 'const', 'fn', 'return', 'if', 'else', 'elif', 'for', 'while',
    'break', 'continue', 'match', 'case', 'yield', 'async', 'await',
    'try', 'catch', 'finally', 'throw', 'with', 'class', 'type',
    'import', 'from', 'as', 'self', 'super', 'inherits',
    'assert', 'pass', 'del', 'is', 'in', 'and', 'or', 'not',
    'true', 'false', 'none',
]

_DSL_KEYWORDS = [
    'reg', 'pair', 'ring', 'evolve', 'OBSERVE', 'observe', 'run',
    'couple', 'entity', 'world', 'sequence', 'via', 'each_for',
    'substrate', 'composed_of',
]

_BUILTINS = [
    'print', 'str', 'int', 'float', 'len', 'range', 'type', 'abs',
    'max', 'min', 'sqrt', 'sum', 'sorted', 'list', 'dict', 'set',
    'tuple', 'map', 'filter', 'enumerate', 'zip', 'open', 'round',
    'isinstance', 'hasattr', 'getattr', 'setattr', 'input',
]

_STDLIB_MODULES = [
    'math', 'random', 'json', 'os', 'sys', 'time', 'pathlib',
    'collections', 'itertools', 'functools', 'datetime', 're',
    'hashlib', 'io', 'csv', 'typing',
    'numpy', 'scipy', 'matplotlib',
]

_TRIAD_MODULES = [
    'nn', 'trainer', 'physics_ml', 'forward',
]

_REGIMES = [
    'B0', 'dispersive', 'anti_collapse', 'R5_crystal', '_pure',
    'B0_3d', 'B0_2d', 'memory_heavy', 'HodgkinHuxley', 'ENSO_recharge',
    'England_autopoietic', 'Eigen_hypercycle', 'Belousov_Zhabotinsky',
    'LSV_market', 'MaxwellWiechert', 'Cepheid_pulsator',
    'DarkMatter_halo', 'Cosmological_inflation',
]

_MATH_MEMBERS = [
    'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'exp', 'log', 'log10', 'log2', 'pow', 'pi', 'e', 'inf', 'nan',
    'ceil', 'floor', 'fabs', 'fmod', 'copysign', 'hypot',
]

_NN_MEMBERS = [
    'triad', 'Conv1d', 'Embedding', 'LayerNorm', 'Dropout',
    'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'GELU', 'SiLU',
    'LeakyReLU', 'ELU', 'MaxPool1d', 'MaxPool2d', 'AvgPool1d',
    'AvgPool2d', 'AdaptiveAvgPool1d', 'GlobalAvgPool',
    'RNN', 'LSTM', 'GRU', 'MultiheadAttention',
    'GroupNorm', 'RMSNorm', 'ResidualBlock', 'CrossAttention',
    'TransformerDecoderBlock', 'AdamW',
    'Sequential', 'Module',
]

_PHYSICS_ML_MEMBERS = [
    'FourierFeatures', 'SpectralConv1d', 'FourierNeuralOperator',
    'HamiltonianLayer', 'PhysicsInformedMLP', 'TriadNeuralOperator',
    'TriadResidualLoss', 'ConservationLoss', 'BoundaryLoss',
]

_TRAINER_MEMBERS = [
    'CosineAnnealingLR', 'WarmupCosineScheduler', 'OneCycleLR',
    'LambdaScheduler', 'GradientClipping', 'GradientAccumulation',
    'SWA', 'ModelCheckpoint', 'MixedPrecision',
]

_DOT_COMPLETIONS = {
    'math': _MATH_MEMBERS,
    'nn': _NN_MEMBERS,
    'physics_ml': _PHYSICS_ML_MEMBERS,
    'trainer': _TRAINER_MEMBERS,
}

def get_completions(prefix: str, source: str = '') -> list[str]:
    if not prefix:
        return []
    if prefix.endswith('.'):
        parts = prefix[:-1].split('.')[-1]
        members = _DOT_COMPLETIONS.get(parts, [])
        return members
    base = prefix.split('.')[-1]
    all_words = _KEYWORDS + _DSL_KEYWORDS + _BUILTINS + _STDLIB_MODULES + _TRIAD_MODULES + _REGIMES
    return sorted(w for w in all_words if w.startswith(base) and w != base)
