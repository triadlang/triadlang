from __future__ import annotations
from adapters import ADAPTERS

_FRAMEWORK_RULES = [
    (['flask'], 'flask'),
]

def detect_framework(tri_file: str) -> list[str]:
    with open(tri_file) as f:
        src = f.read()
    imports = _extract_imports(src)
    detected = []
    for required, adapter_name in _FRAMEWORK_RULES:
        if any(imp in imports for imp in required):
            detected.append(adapter_name)
    return detected

def _extract_imports(src: str) -> set[str]:
    imports = set()
    for line in src.splitlines():
        stripped = line.strip().rstrip(';').rstrip()
        if stripped.startswith('import '):
            parts = stripped[len('import '):].split()
            if parts:
                imports.add(parts[0].split('.')[0])
        elif stripped.startswith('from '):
            parts = stripped[len('from '):].split()
            if parts:
                imports.add(parts[0].split('.')[0])
    return imports

def auto_load(tri_file: str, **kwargs):
    detected = detect_framework(tri_file)
    if not detected:
        adapter_cls = ADAPTERS['headless']
        return adapter_cls(tri_file=tri_file, **kwargs)
    adapter_name = detected[0]
    if adapter_name not in ADAPTERS:
        available = ', '.join(sorted(ADAPTERS.keys()))
        raise ValueError(
            f"detected framework '{adapter_name}' but no adapter available. "
            f"available adapters: {available}. "
            f"to add one, create a new adapter in src/adapters/ and register it."
        )
    adapter_cls = ADAPTERS[adapter_name]
    return adapter_cls(tri_file=tri_file, **kwargs)
