from __future__ import annotations

import os
from abc import ABC, abstractmethod

from embed.engine import TriadEngine
from runtime.security import default_policy


class BaseAdapter(ABC):

    def __init__(self, tri_file: str, search_paths: list[str] | None = None,
                 backend: str = 'auto', safe: bool = True):
        project_dir = os.path.dirname(os.path.abspath(tri_file)) or os.getcwd()
        self._engine = TriadEngine(
            search_paths=search_paths, backend=backend, safe=safe,
            policy=default_policy(project_dir),
        )
        self._tri_file = tri_file
        self._setup()

    @abstractmethod
    def _setup(self):
        pass

    @abstractmethod
    def start(self):
        pass
