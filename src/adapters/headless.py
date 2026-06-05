from __future__ import annotations
from adapters.base import BaseAdapter

class HeadlessAdapter(BaseAdapter):

    def _setup(self):
        self._result = None

    def start(self):
        self._result = self._engine.run_file(self._tri_file)
        return self._result

    def get(self, name: str):
        return self._engine.get(name)
