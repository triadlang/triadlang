from pathlib import Path
from zipfile import ZipFile

import pytest


def test_built_wheel_contains_runtime_resources():
    wheels = sorted(Path('dist').glob('triadlang-*.whl'))
    if not wheels:
        pytest.skip('wheel has not been built')

    with ZipFile(wheels[-1]) as archive:
        names = set(archive.namelist())

    required = {
        'api/dashboard.html',
        'api/engine.html',
        'stdlib/triad/kernel.tri',
        'stdlib/triad/solver_bridge.tri',
    }
    assert required <= names
