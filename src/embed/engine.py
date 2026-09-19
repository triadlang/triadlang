from __future__ import annotations

import os

from frontend.parser_universal import parse
from runtime.compiler_runtime import TriadCompiler
from runtime.security import SecurityPolicy, get_policy


class TriadEngine:

    def __init__(self, search_paths: list[str] | None = None,
                 backend: str = 'auto', safe: bool = True,
                 policy: SecurityPolicy | None = None):
        self._search_paths = search_paths or ['.']
        self._backend = backend
        self._safe = safe
        self._policy_obj = policy
        self._extra_globals: dict = {}
        self._last_namespace: dict = {}

    def _policy(self) -> SecurityPolicy:
        if self._policy_obj is not None:
            return self._policy_obj
        return get_policy()

    def register(self, name: str, obj) -> None:
        self._extra_globals[name] = obj

    def get(self, name: str):
        return self._last_namespace.get(name)

    def run_source(self, text: str, file: str = '<triad>') -> dict:
        policy = self._policy()
        if policy is not None and policy.safe != self._safe:
            policy = SecurityPolicy(safe=self._safe, sandbox=policy.sandbox, capabilities=policy.capabilities)
        mod = parse(text, file)
        compiler = TriadCompiler()
        compiler._search_paths = self._search_paths
        code = compiler.compile_to_source(mod)
        env = compiler._make_globals(file, safe=self._safe)
        env['_triad_policy'] = policy
        env.update(self._extra_globals)
        env['_search_paths'] = compiler._search_paths
        compiled = compile(code, file, 'exec')
        exec(compiled, env)
        self._last_namespace = {
            k: v for k, v in env.items()
            if not k.startswith('__')
        }
        return self._last_namespace

    def run_file(self, path: str) -> dict:
        path = os.path.abspath(path)
        policy = self._policy()
        if policy.safe:
            policy.sandbox.resolve(path, mode='read', must_exist=True)
        with open(path) as f:
            src = f.read()
        return self.run_source(src, file=path)

    def eval_expr(self, text: str):

        expr = text.strip()
        while expr.endswith(';'):
            expr = expr[:-1].rstrip()
        ns = self.run_source(f'let _triad_eval = ({expr});')
        return ns.get('_triad_eval')
