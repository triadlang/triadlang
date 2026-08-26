from __future__ import annotations

import os
from pathlib import Path


class PathViolation(Exception):
    pass


class TriadSandbox:
    def __init__(self, workspaces: list[str] | None = None, allow_temp: bool = True,
                 allow_home_read: set[str] | None = None):
        self.workspaces: list[Path] = []
        for w in (workspaces or ['.']):
            p = Path(w).resolve()
            self.workspaces.append(p)
        self.allow_temp = allow_temp
        self.allow_home_read = set(allow_home_read or [])
        self._temp_root = Path(os.environ.get('TMPDIR') or '/tmp').resolve()

    def resolve(self, path: str, mode: str = 'read', must_exist: bool = False) -> Path:
        p = Path(path)
        if not p.is_absolute():
            if self.workspaces:
                base = self.workspaces[0]
            else:
                base = Path('.').resolve()
            p = (base / p).resolve()
        else:
            p = p.resolve()
        self._check(p, mode)
        if must_exist and not p.exists():
            raise FileNotFoundError(p)
        return p

    def _check(self, p: Path, mode: str):
        try:
            p = p.resolve()
        except (OSError, ValueError) as exc:
            raise PathViolation(f'invalid path: {p}') from exc
        if self._inside_any(p, self.workspaces):
            return
        if self.allow_temp and self._inside_any(p, [self._temp_root]):
            return
        if mode == 'read' and self._home_allowed(p):
            return
        raise PathViolation(f'path {p} outside allowed workspaces')

    def _inside_any(self, p: Path, roots: list[Path]) -> bool:
        for r in roots:
            try:
                p.relative_to(r)
                return True
            except ValueError:
                pass
        return False

    def _home_allowed(self, p: Path) -> bool:
        home = Path.home()
        try:
            p.relative_to(home)
        except ValueError:
            return False
        for allowed in self.allow_home_read:
            if p == home / allowed or self._inside_any(p, [home / allowed]):
                return True
        return False

    def read_text(self, path: str) -> str:
        p = self.resolve(path, mode='read', must_exist=True)
        return p.read_text(encoding='utf-8')

    def read_bytes(self, path: str) -> bytes:
        p = self.resolve(path, mode='read', must_exist=True)
        return p.read_bytes()

    def write_text(self, path: str, content: str):
        p = self.resolve(path, mode='write')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')

    def write_bytes(self, path: str, data: bytes):
        p = self.resolve(path, mode='write')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def exists(self, path: str) -> bool:
        try:
            p = self.resolve(path, mode='read')
            return p.exists()
        except PathViolation:
            return False

    def listdir(self, path: str) -> list[str]:
        p = self.resolve(path, mode='read', must_exist=True)
        if not p.is_dir():
            raise NotADirectoryError(p)
        return sorted(str(c) for c in p.iterdir())

    def remove(self, path: str):
        p = self.resolve(path, mode='write', must_exist=True)
        if p.is_dir():
            import shutil
            shutil.rmtree(p)
        else:
            p.unlink()

    def copy(self, src: str, dst: str):
        src_p = self.resolve(src, mode='read', must_exist=True)
        dst_p = self.resolve(dst, mode='write')
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        if src_p.is_dir():
            import shutil
            shutil.copytree(src_p, dst_p, dirs_exist_ok=True)
        else:
            import shutil
            shutil.copy2(src_p, dst_p)

    def mkdir(self, path: str):
        p = self.resolve(path, mode='write')
        p.mkdir(parents=True, exist_ok=True)


def default_sandbox(project_dir: str = '.') -> TriadSandbox:
    return TriadSandbox(workspaces=[project_dir], allow_temp=True)
