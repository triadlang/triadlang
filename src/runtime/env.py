from __future__ import annotations

import os
import threading

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))

_ENV_FILE: dict | None = None
_ENV_LOCK = threading.Lock()

def _load_env_file() -> dict:
    global _ENV_FILE
    with _ENV_LOCK:
        if _ENV_FILE is None:
            _ENV_FILE = {}
            path = os.path.join(REPO_ROOT, '.env')
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        if line.startswith('export '):
                            line = line[len('export '):].lstrip()
                        k, v = line.split('=', 1)
                        v = v.strip()
                        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                            v = v[1:-1]
                        _ENV_FILE[k.strip()] = v
        return _ENV_FILE

def load_os_environ():
    for k, v in _load_env_file().items():
        os.environ.setdefault(k, v)

def env(key: str, default):
    raw = os.environ.get(key)
    if raw is None:
        raw = _load_env_file().get(key)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() not in ('0', 'false', 'no', 'off', '')
    try:
        return type(default)(raw)
    except (ValueError, TypeError) as e:
        raise ValueError(f'env {key}={raw!r} invalid (expected {type(default).__name__}): {e}')

