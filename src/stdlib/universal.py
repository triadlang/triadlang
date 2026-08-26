from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
import re as _re
import shutil as _shutil
import time as _time
import urllib.error as _uerror
import urllib.parse as _uparse
import urllib.request as _urequest
from pathlib import Path as _Path

try:
    from runtime.security import get_policy
except Exception:
    get_policy = None


def _policy():
    if get_policy is not None:
        return get_policy()
    return None

def re_match(pattern: str, text: str):
    m = _re.match(pattern, text)
    return m.group(0) if m else None

def re_search(pattern: str, text: str):
    m = _re.search(pattern, text)
    return m.group(0) if m else None

def re_find_all(pattern: str, text: str) -> list:
    return _re.findall(pattern, text)

def re_replace(pattern: str, repl: str, text: str, count: int = 0) -> str:
    return _re.sub(pattern, repl, text, count=count)

def re_split(pattern: str, text: str) -> list:
    return _re.split(pattern, text)

def re_groups(pattern: str, text: str):
    m = _re.search(pattern, text)
    if m is None:
        return None
    return list(m.groups()) if m.groups() else [m.group(0)]

RE_EXPORTS = {
    'match': re_match, 'search': re_search, 'find_all': re_find_all,
    'replace': re_replace, 'split': re_split, 'groups': re_groups,
}

def fs_read(path: str) -> str:
    p = _policy()
    if p is not None and p.safe:
        return p.sandbox.read_text(path)
    return _Path(path).read_text(encoding='utf-8')

def fs_write(path: str, content: str):
    p = _policy()
    if p is not None and p.safe:
        p.sandbox.write_text(path, content)
        return
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    _Path(path).write_text(content, encoding='utf-8')

def fs_append(path: str, content: str):
    p = _policy()
    if p is not None and p.safe:
        p.sandbox.resolve(path, mode='write')
    p2 = _Path(path)
    p2.parent.mkdir(parents=True, exist_ok=True)
    with open(p2, 'a', encoding='utf-8') as f:
        f.write(content)

def fs_read_bytes(path: str) -> bytes:
    p = _policy()
    if p is not None and p.safe:
        return p.sandbox.read_bytes(path)
    return _Path(path).read_bytes()

def fs_write_bytes(path: str, data: bytes):
    p = _policy()
    if p is not None and p.safe:
        p.sandbox.write_bytes(path, data)
        return
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    _Path(path).write_bytes(data)

def fs_read_json(path: str):
    return _json.loads(fs_read(path))

def fs_write_json(path: str, value, indent: int = 2):
    fs_write(path, _json.dumps(value, indent=indent, ensure_ascii=False))

def fs_exists(path: str) -> bool:
    p = _policy()
    if p is not None and p.safe:
        return p.sandbox.exists(path)
    return _os.path.exists(path)

def fs_is_file(path: str) -> bool:
    p = _policy()
    if p is not None and p.safe:
        try:
            return p.sandbox.resolve(path, mode='read').is_file()
        except Exception:
            return False
    return _os.path.isfile(path)

def fs_is_dir(path: str) -> bool:
    p = _policy()
    if p is not None and p.safe:
        try:
            return p.sandbox.resolve(path, mode='read').is_dir()
        except Exception:
            return False
    return _os.path.isdir(path)

def fs_list(path: str = '.', pattern: str = '*') -> list:
    p = _policy()
    if p is not None and p.safe:
        base = p.sandbox.resolve(path, mode='read', must_exist=True)
        return sorted(str(c) for c in base.glob(pattern) if c.is_file())
    return sorted(str(p) for p in _Path(path).glob(pattern))

def fs_walk(path: str = '.', pattern: str = '**/*') -> list:
    p = _policy()
    if p is not None and p.safe:
        base = p.sandbox.resolve(path, mode='read', must_exist=True)
        return sorted(str(c) for c in base.glob(pattern) if c.is_file())
    return sorted(str(p) for p in _Path(path).glob(pattern) if p.is_file())

def fs_mkdir(path: str):
    p = _policy()
    if p is not None and p.safe:
        p.sandbox.mkdir(path)
        return
    _os.makedirs(path, exist_ok=True)

def fs_remove(path: str):
    p = _policy()
    if p is not None and p.safe:
        p.require_capability('fs.remove')
        p.sandbox.remove(path)
        return
    if _os.path.isdir(path):
        _shutil.rmtree(path)
    elif _os.path.exists(path):
        _os.remove(path)

def fs_copy(src: str, dst: str):
    p = _policy()
    if p is not None and p.safe:
        p.sandbox.copy(src, dst)
        return
    if _os.path.isdir(src):
        _shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        _Path(dst).parent.mkdir(parents=True, exist_ok=True)
        _shutil.copy2(src, dst)

def fs_move(src: str, dst: str):
    p = _policy()
    if p is not None and p.safe:
        p.require_capability('fs.remove')
        p.sandbox.copy(src, dst)
        p.sandbox.remove(src)
        return
    _shutil.move(src, dst)

def fs_size(path: str) -> int:
    p = _policy()
    if p is not None and p.safe:
        return p.sandbox.resolve(path, mode='read', must_exist=True).stat().st_size
    return _os.path.getsize(path)

def fs_join(*parts) -> str:
    return _os.path.join(*parts)

def fs_basename(path: str) -> str:
    return _os.path.basename(path)

def fs_dirname(path: str) -> str:
    return _os.path.dirname(path)

def fs_absolute(path: str) -> str:
    p = _policy()
    if p is not None and p.safe:
        return str(p.sandbox.resolve(path, mode='read'))
    return _os.path.abspath(path)

def fs_cwd() -> str:
    p = _policy()
    if p is not None and p.safe:
        return str(p.sandbox.workspaces[0]) if p.sandbox.workspaces else _os.getcwd()
    return _os.getcwd()

def fs_home() -> str:
    return str(_Path.home())

FS_EXPORTS = {
    'read': fs_read, 'write': fs_write, 'append': fs_append,
    'read_bytes': fs_read_bytes, 'write_bytes': fs_write_bytes,
    'read_json': fs_read_json, 'write_json': fs_write_json,
    'exists': fs_exists, 'is_file': fs_is_file, 'is_dir': fs_is_dir,
    'list': fs_list, 'walk': fs_walk, 'mkdir': fs_mkdir,
    'remove': fs_remove, 'copy': fs_copy, 'move': fs_move,
    'size': fs_size, 'join': fs_join, 'basename': fs_basename,
    'dirname': fs_dirname, 'absolute': fs_absolute,
    'cwd': fs_cwd, 'home': fs_home,
}

class HttpResponse:
    def __init__(self, status: int, headers: dict, body: bytes, url: str):
        self.status = status
        self.headers = headers
        self.body = body
        self.url = url

    @property
    def text(self) -> str:
        return self.body.decode('utf-8', errors='replace')

    def json(self):
        return _json.loads(self.text)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def __repr__(self):
        return f'HttpResponse(status={self.status}, url={self.url!r}, bytes={len(self.body)})'

def _http_request(method: str, url: str, data=None, headers=None,
                  timeout: float = 30.0, params=None) -> HttpResponse:
    if params:
        sep = '&' if '?' in url else '?'
        url = url + sep + _uparse.urlencode(params)
    body = None
    hdrs = dict(headers or {})
    if data is not None:
        if isinstance(data, (dict, list)):
            body = _json.dumps(data).encode('utf-8')
            hdrs.setdefault('Content-Type', 'application/json')
        elif isinstance(data, str):
            body = data.encode('utf-8')
        else:
            body = bytes(data)
    hdrs.setdefault('User-Agent', 'triadlang-http/1.0')
    req = _urequest.Request(url, data=body, headers=hdrs, method=method)
    try:
        with _urequest.urlopen(req, timeout=timeout) as resp:
            return HttpResponse(resp.status, dict(resp.headers),
                                resp.read(), resp.geturl())
    except _uerror.HTTPError as e:
        return HttpResponse(e.code, dict(e.headers or {}),
                            e.read() if hasattr(e, 'read') else b'', url)

def http_get(url: str, headers=None, params=None, timeout: float = 30.0) -> HttpResponse:
    p = _policy()
    if p is not None and p.safe and not p.capabilities.can_http_get(url):
        raise PermissionError(f'http GET {url} not allowed by capabilities')
    return _http_request('GET', url, headers=headers, params=params, timeout=timeout)

def http_post(url: str, data=None, headers=None, timeout: float = 30.0) -> HttpResponse:
    p = _policy()
    if p is not None and p.safe and not p.capabilities.can_http_post(url):
        raise PermissionError(f'http POST {url} not allowed by capabilities')
    return _http_request('POST', url, data=data, headers=headers, timeout=timeout)

def http_put(url: str, data=None, headers=None, timeout: float = 30.0) -> HttpResponse:
    p = _policy()
    if p is not None and p.safe and not p.capabilities.net_http_other:
        raise PermissionError(f'http PUT {url} not allowed by capabilities')
    return _http_request('PUT', url, data=data, headers=headers, timeout=timeout)

def http_delete(url: str, headers=None, timeout: float = 30.0) -> HttpResponse:
    p = _policy()
    if p is not None and p.safe and not p.capabilities.net_http_other:
        raise PermissionError(f'http DELETE {url} not allowed by capabilities')
    return _http_request('DELETE', url, headers=headers, timeout=timeout)

def http_download(url: str, path: str, timeout: float = 60.0) -> int:
    p = _policy()
    if p is not None and p.safe:
        if not p.capabilities.can_http_get(url):
            raise PermissionError(f'http GET {url} not allowed by capabilities')
        p.sandbox.resolve(path, mode='write')
    r = http_get(url, timeout=timeout)
    fs_write_bytes(path, r.body)
    return len(r.body)

HTTP_EXPORTS = {
    'get': http_get, 'post': http_post, 'put': http_put,
    'delete': http_delete, 'download': http_download,
    'url_encode': _uparse.urlencode, 'url_quote': _uparse.quote,
    'HttpResponse': HttpResponse,
}

def time_now() -> float:
    return _time.time()

def time_monotonic() -> float:
    return _time.monotonic()

def time_sleep(seconds: float):
    _time.sleep(seconds)

def time_stamp(fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
    return _dt.datetime.now().strftime(fmt)

def time_utc(fmt: str = '%Y-%m-%dT%H:%M:%SZ') -> str:
    return _dt.datetime.now(_dt.UTC).strftime(fmt)

def time_parse(text: str, fmt: str = '%Y-%m-%d %H:%M:%S') -> float:
    return _dt.datetime.strptime(text, fmt).timestamp()

def time_format(epoch: float, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
    return _dt.datetime.fromtimestamp(epoch).strftime(fmt)

def time_diff_days(epoch_a: float, epoch_b: float) -> float:
    return (epoch_b - epoch_a) / 86400.0

def time_add_days(epoch: float, days: float) -> float:
    return epoch + days * 86400.0

def time_weekday(epoch: float | None = None) -> str:
    d = _dt.datetime.fromtimestamp(epoch) if epoch is not None else _dt.datetime.now()
    return d.strftime('%A')

TIME_EXPORTS = {
    'now': time_now, 'monotonic': time_monotonic, 'sleep': time_sleep,
    'stamp': time_stamp, 'utc': time_utc, 'parse': time_parse,
    'format': time_format, 'diff_days': time_diff_days,
    'add_days': time_add_days, 'weekday': time_weekday,
}
