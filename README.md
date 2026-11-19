# TriadLang Python

This directory contains the Python implementation of TriadLang:

- `cli/`, `frontend/`, `compiler/`, `runtime/`, `stdlib/`
- `tests/`, `examples/`, `benchmarks/`, `scripts/`
- `triad`, `triadc`, `entry.py`, and `build_dist.sh`

From the repo root:

```bash
./triad run python/examples/basic/hello.tri
python -m pytest
```

The root `pyproject.toml` installs Python packages from this directory.
