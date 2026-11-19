from adapters.base import BaseAdapter
from adapters.headless import HeadlessAdapter
from adapters.flask import FlaskAdapter

ADAPTERS = {
    'headless': HeadlessAdapter,
    'flask': FlaskAdapter,
}

def get_adapter(name: str):
    if name not in ADAPTERS:
        available = ', '.join(sorted(ADAPTERS.keys()))
        raise ValueError(
            f"unknown adapter '{name}'. available: {available}"
        )
    return ADAPTERS[name]
