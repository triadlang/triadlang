from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field


@dataclass
class CapabilitySet:
    fs_read: list[str] = field(default_factory=list)
    fs_write: list[str] = field(default_factory=list)
    fs_remove: bool = False
    net_http_get: list[str] = field(default_factory=list)
    net_http_post: list[str] = field(default_factory=list)
    net_http_other: bool = False
    net_socket: bool = False
    env_read: bool = False
    env_write: bool = False
    process_spawn: bool = False
    process_exec: bool = False
    ml_load_local: list[str] = field(default_factory=list)
    ml_download_remote: bool = False
    py_native: bool = False
    ccall: bool = False
    kernel_call: bool = False

    def can_read(self, path: str) -> bool:
        for p in self.fs_read:
            if fnmatch.fnmatch(path, p):
                return True
        return False

    def can_write(self, path: str) -> bool:
        for p in self.fs_write:
            if fnmatch.fnmatch(path, p):
                return True
        return False

    def can_http_get(self, url: str) -> bool:
        for p in self.net_http_get:
            if fnmatch.fnmatch(url, p):
                return True
        return False

    def can_http_post(self, url: str) -> bool:
        for p in self.net_http_post:
            if fnmatch.fnmatch(url, p):
                return True
        return False

    def can_load_model(self, path: str) -> bool:
        for p in self.ml_load_local:
            if fnmatch.fnmatch(path, p):
                return True
        return False


def parse_capabilities(source: str) -> CapabilitySet:
    cap = CapabilitySet()
    if not source:
        return cap
    for raw in source.split(';'):
        raw = raw.strip()
        if not raw:
            continue
        if ':' not in raw:
            key = raw
            val = ''
        else:
            key, val = raw.split(':', 1)
            key = key.strip()
            val = val.strip()
        if key == 'fs.read':
            cap.fs_read.extend([v.strip() for v in val.split(',') if v.strip()])
        elif key == 'fs.write':
            cap.fs_write.extend([v.strip() for v in val.split(',') if v.strip()])
        elif key == 'fs.remove':
            cap.fs_remove = val.lower() in ('1', 'true', 'yes')
        elif key == 'net.get':
            cap.net_http_get.extend([v.strip() for v in val.split(',') if v.strip()])
        elif key == 'net.post':
            cap.net_http_post.extend([v.strip() for v in val.split(',') if v.strip()])
        elif key == 'net.other':
            cap.net_http_other = val.lower() in ('1', 'true', 'yes')
        elif key == 'net.socket':
            cap.net_socket = val.lower() in ('1', 'true', 'yes')
        elif key == 'env.read':
            cap.env_read = val.lower() in ('1', 'true', 'yes')
        elif key == 'env.write':
            cap.env_write = val.lower() in ('1', 'true', 'yes')
        elif key == 'process.spawn':
            cap.process_spawn = val.lower() in ('1', 'true', 'yes')
        elif key == 'process.exec':
            cap.process_exec = val.lower() in ('1', 'true', 'yes')
        elif key == 'ml.local':
            cap.ml_load_local.extend([v.strip() for v in val.split(',') if v.strip()])
        elif key == 'ml.remote':
            cap.ml_download_remote = val.lower() in ('1', 'true', 'yes')
        elif key == 'py.native':
            cap.py_native = val.lower() in ('1', 'true', 'yes')
        elif key == 'ccall':
            cap.ccall = val.lower() in ('1', 'true', 'yes')
        elif key == 'kernel.call':
            cap.kernel_call = val.lower() in ('1', 'true', 'yes')
    return cap
