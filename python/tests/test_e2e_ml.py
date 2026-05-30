import pytest
import os
import numpy as np
from runtime.compiler_runtime import TriadCompiler
from frontend.parser_universal import parse
TRI_SOURCE = '\nimport triad.tensor as T\nimport triad.nn as nn\nimport triad.losses as losses\n\nlet model = nn.Sequential(\n    nn.Linear(2, 8),\n    nn.ReLU(),\n    nn.Linear(8, 1),\n    nn.Sigmoid()\n);\n\nlet opt = nn.Adam(model.parameters(), 0.05);\n\nlet x = T.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);\nlet y = T.tensor([[0.0], [1.0], [1.0], [0.0]]);\n\nfor i in range(500) {\n    let pred = model(x);\n    let loss = losses.bce_loss(pred, y);\n    opt.zero_grad();\n    loss.backward();\n    opt.step();\n}\n\nlet final = model(x);\nprint(final[0]._data);\nprint(final[1]._data);\nprint(final[2]._data);\nprint(final[3]._data);\n'

class TestE2ETraining:

    def test_xor_from_source(self, capsys):
        ast = parse(TRI_SOURCE, '<test_xor.tri>')
        compiler = TriadCompiler()
        compiler.compile_and_run(ast)
        captured = capsys.readouterr()
        lines = [l.strip() for l in captured.out.strip().split('\n') if l.strip()]
        assert len(lines) == 4
        p0 = float(lines[0].strip('[]'))
        p1 = float(lines[1].strip('[]'))
        p2 = float(lines[2].strip('[]'))
        p3 = float(lines[3].strip('[]'))
        assert p0 < 0.3
        assert p1 > 0.7
        assert p2 > 0.7
        assert p3 < 0.3

    def test_xor_from_file(self, capsys, tmp_path):
        tri_file = tmp_path / 'test.tri'
        tri_file.write_text(TRI_SOURCE)
        ast = parse(tri_file.read_text(), str(tri_file))
        ast.file = str(tri_file)
        compiler = TriadCompiler()
        compiler.compile_and_run(ast)
        captured = capsys.readouterr()
        lines = [l.strip() for l in captured.out.strip().split('\n') if l.strip()]
        assert len(lines) == 4
TRI_DATA_PIPELINE = '\nimport triad.tensor as T\nimport triad.nn as nn\nimport triad.data as data\n\nlet x = T.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],\n                   [2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 1.0, 2.0],\n                   [1.5, 0.5, 0.0], [0.5, 1.5, 0.0], [0.0, 0.5, 1.5],\n                   [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0],\n                   [2.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 2.0, 1.0],\n                   [3.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 3.0],\n                   [1.0, 1.0, 1.0], [2.0, 2.0, 1.0]]);\nlet y = T.tensor([[3.0], [2.0], [1.0], [5.0], [4.0], [3.0],\n                   [4.0], [3.0], [2.0], [3.0], [2.0], [3.0],\n                   [5.0], [4.0], [4.0], [7.0], [5.0], [5.0],\n                   [4.0], [7.0]]);\n\nlet ds = data.Dataset(x, y);\nlet train_ds = ds[0:16];\nlet val_ds = ds[16:20];\n\nlet train_dl = data.DataLoader(train_ds, 8, true);\nlet val_dl = data.DataLoader(val_ds, 4, false);\n\nlet model = nn.Linear(3, 1);\nlet opt = nn.Adam(model.parameters(), 0.05);\n\nfor epoch in range(100) {\n    for batch in train_dl {\n        let bx = batch[0];\n        let by = batch[1];\n        let pred = model(bx);\n        let diff = pred - by;\n        let loss = (diff * diff).mean();\n        opt.zero_grad();\n        loss.backward();\n        opt.step();\n    }\n}\n\nlet test_x = T.tensor([[1.0, 0.0, 0.0]]);\nlet test_y = model(test_x);\nprint(test_y._data);\n'

class TestE2EDataPipeline:

    def test_data_pipeline(self, capsys):
        np.random.seed(42)
        ast = parse(TRI_DATA_PIPELINE, '<test_data.tri>')
        compiler = TriadCompiler()
        compiler.compile_and_run(ast)
        captured = capsys.readouterr()
        val = float(captured.out.strip().strip('[]'))
        assert abs(val - 3.0) < 1.0
TRI_SAVE_LOAD = '\nimport triad.tensor as T\nimport triad.nn as nn\nimport triad.train as train\n\nlet model = nn.Linear(4, 2);\nlet w_copy = T.tensor(model.weight._data.copy());\n\ntrain.save_weights(model, "/tmp/triad_test_weights.json");\n\nmodel.weight._data[0][0] = 0.0;\nmodel.weight._data[0][1] = 0.0;\n\ntrain.load_weights(model, "/tmp/triad_test_weights.json");\n\nlet diff = model.weight - w_copy;\nlet ok = (diff * diff).sum();\nprint(ok._data);\n'

class TestE2ESerialization:

    def test_save_load_from_tri(self, capsys):
        ast = parse(TRI_SAVE_LOAD, '<test_serialize.tri>')
        compiler = TriadCompiler()
        compiler.compile_and_run(ast)
        captured = capsys.readouterr()
        val = float(captured.out.strip())
        assert val < 1e-10