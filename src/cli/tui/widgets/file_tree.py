from __future__ import annotations

import os

from textual.message import Message
from textual.widgets import DirectoryTree, Static


class FileSelected(Message):

    def __init__(self, path: str):
        super().__init__()
        self.path = path

class TriadFileTree(Static):

    DEFAULT_CSS = """
    TriadFileTree {
        height: 1fr;
        overflow-y: auto;
    }
    TriadFileTree DirectoryTree {
        height: 1fr;
    }
    """

    def __init__(self, path=".", **kw):
        super().__init__(**kw)
        self._root = os.path.abspath(path)

    def compose(self):
        yield DirectoryTree(self._root, id="file-tree-inner")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:

        path = str(event.path)
        if path.endswith(".tri"):
            self.post_message(FileSelected(path))

