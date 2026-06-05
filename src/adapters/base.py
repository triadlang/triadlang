from __future__ import annotations
from abc import ABC, abstractmethod
from embed.engine import TriadEngine

class BaseAdapter(ABC):

    def __init__(self, tri_file: str, search_paths: list[str] | None = None,
                 backend: str = 'auto'):
        self._engine = TriadEngine(search_paths=search_paths, backend=backend)
        self._tri_file = tri_file
        self._setup()

    @abstractmethod
    def _setup(self):
        pass

    @abstractmethod
    def start(self):
        pass
