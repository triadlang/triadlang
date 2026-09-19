from __future__ import annotations

import os
import re

from fastapi import APIRouter, HTTPException

router = APIRouter()

_EXAMPLES_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples')
)

_NAME_RE = re.compile(r'^[A-Za-z0-9_\-\.]+$')

def _safe_name(name: str) -> str:
    if not _NAME_RE.match(name):
        raise HTTPException(status_code=400, detail='invalid template name')
    return name

def _collect_templates() -> list[dict]:
    templates = []
    if not os.path.isdir(_EXAMPLES_ROOT):
        return templates
    for category in sorted(os.listdir(_EXAMPLES_ROOT)):
        cat_dir = os.path.join(_EXAMPLES_ROOT, category)
        if not os.path.isdir(cat_dir) or category.startswith('__'):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith('.tri') or fname.startswith(('.', '__')):
                continue
            fpath = os.path.join(cat_dir, fname)
            try:
                with open(fpath) as f:
                    content = f.read()
            except OSError:
                content = ''
            templates.append({
                'category': category,
                'name': fname,
                'path': f'examples/{category}/{fname}',
                'source': content,
            })
    return templates

@router.get('/', summary='List all example .tri templates grouped by category')
async def list_templates(category: str | None = None):
    all_templates = _collect_templates()
    if category:
        all_templates = [t for t in all_templates if t['category'] == category]
    categories = sorted(set(t['category'] for t in all_templates))
    return {
        'templates': all_templates,
        'categories': categories,
    }

@router.get('/{category}/{name}', summary='Get a single template source')
async def get_template(category: str, name: str):
    category = _safe_name(category)
    name = _safe_name(name)
    fpath = os.path.join(_EXAMPLES_ROOT, category, name)
    fpath = os.path.abspath(fpath)
    if not fpath.startswith(_EXAMPLES_ROOT) or not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail=f'template not found: {category}/{name}')
    with open(fpath) as f:
        source = f.read()
    return {
        'category': category,
        'name': name,
        'path': f'examples/{category}/{name}',
        'source': source,
    }
