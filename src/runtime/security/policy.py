from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar, Token

from runtime.security.capabilities import CapabilitySet, parse_capabilities
from runtime.security.sandbox import TriadSandbox, default_sandbox


class SecurityPolicy:
    def __init__(self, safe: bool = True, sandbox: TriadSandbox | None = None,
                 capabilities: CapabilitySet | None = None):
        self.safe = safe
        self.sandbox = sandbox or default_sandbox()
        self.capabilities = capabilities or CapabilitySet()

    def require_safe(self):
        if not self.safe:
            raise PermissionError('operation only allowed in safe mode')

    def require_unsafe(self, feature: str):
        if self.safe:
            raise PermissionError(
                f'{feature} is disabled in safe mode. use --unsafe to enable it.')

    _KNOWN_CAPABILITIES = frozenset({
        'py.native', 'ccall', 'process.exec', 'process.spawn',
        'net.http_get', 'net.http_post', 'net.socket',
        'env.read', 'env.write', 'fs.remove', 'ml.remote',
    })

    def require_capability(self, name: str, detail: str = ''):
        if name not in self._KNOWN_CAPABILITIES:
            raise PermissionError(f'unknown capability {name!r}')
        if name == 'py.native' and not self.capabilities.py_native:
            raise PermissionError('py_native capability not granted' +
                                  (f': {detail}' if detail else ''))
        if name == 'ccall' and not self.capabilities.ccall:
            raise PermissionError('ccall capability not granted' +
                                  (f': {detail}' if detail else ''))
        if name == 'process.exec' and not self.capabilities.process_exec:
            raise PermissionError('process.exec capability not granted')
        if name == 'process.spawn' and not self.capabilities.process_spawn:
            raise PermissionError('process.spawn capability not granted')
        if name == 'net.http_get' and not self.capabilities.net_http_get:
            raise PermissionError('net.http_get capability not granted')
        if name == 'net.http_post' and not self.capabilities.net_http_post:
            raise PermissionError('net.http_post capability not granted')
        if name == 'net.socket' and not self.capabilities.net_socket:
            raise PermissionError('net.socket capability not granted')
        if name == 'env.read' and not self.capabilities.env_read:
            raise PermissionError('env.read capability not granted')
        if name == 'env.write' and not self.capabilities.env_write:
            raise PermissionError('env.write capability not granted')
        if name == 'fs.remove' and not self.capabilities.fs_remove:
            raise PermissionError('fs.remove capability not granted')
        if name == 'ml.remote' and not self.capabilities.ml_download_remote:
            raise PermissionError('ml.remote_download capability not granted')


_POLICY: ContextVar[SecurityPolicy | None] = ContextVar('triad_security_policy', default=None)


def default_policy(project_dir: str = '.') -> SecurityPolicy:
    cap_source = os.environ.get('TRIAD_CAPABILITIES', '')
    return SecurityPolicy(
        safe=True,
        sandbox=default_sandbox(project_dir),
        capabilities=parse_capabilities(cap_source),
    )


def set_policy(policy: SecurityPolicy) -> Token:
    return _POLICY.set(policy)


def get_policy() -> SecurityPolicy:
    policy = _POLICY.get()
    if policy is None:
        policy = default_policy()
        _POLICY.set(policy)
    return policy


def reset_policy():
    _POLICY.set(None)


@contextmanager
def use_policy(policy: SecurityPolicy):
    token = _POLICY.set(policy)
    try:
        yield policy
    finally:
        _POLICY.reset(token)
