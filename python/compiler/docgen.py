from __future__ import annotations
import os
import re
import sys
from typing import Optional

def parse_tri_docs(source: str) -> dict:
    lines = source.split('\n')
    result = {'module_doc': '', 'functions': [], 'classes': [], 'types': [], 'constants': [], 'imports': []}
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith('//'):
            comment = stripped[2:].strip()
            if not result['module_doc'] and indent == 0:
                result['module_doc'] = comment
            i += 1
            continue
        if indent == 0 and stripped.startswith('fn '):
            fn_info = _parse_fn(lines, i)
            if fn_info:
                result['functions'].append(fn_info)
        if indent == 0 and stripped.startswith('class '):
            cls_info = _parse_class(lines, i)
            if cls_info:
                result['classes'].append(cls_info)
        if indent == 0 and stripped.startswith('type '):
            type_info = _parse_type(lines, i)
            if type_info:
                result['types'].append(type_info)
        if indent == 0 and stripped.startswith('const '):
            const_info = _parse_const(stripped)
            if const_info:
                result['constants'].append(const_info)
        if indent == 0 and (stripped.startswith('import ') or stripped.startswith('from ')):
            result['imports'].append(stripped)
        i += 1
    return result

def _parse_fn(lines: list[str], start: int) -> Optional[dict]:
    line = lines[start].lstrip()
    m = re.match('fn\\s+(\\w+)\\s*\\(([^)]*)\\)', line)
    if not m:
        return None
    name = m.group(1)
    params_raw = m.group(2)
    params = _parse_params(params_raw)
    doc = ''
    if start + 1 < len(lines):
        next_line = lines[start + 1].lstrip()
        if next_line.startswith('//'):
            doc = next_line[2:].strip()
    return {'name': name, 'params': params, 'doc': doc, 'line': start + 1}

def _parse_class(lines: list[str], start: int) -> Optional[dict]:
    line = lines[start].lstrip()
    m = re.match('class\\s+(\\w+)(?:\\s*<\\s*(\\w+))?', line)
    if not m:
        return None
    name = m.group(1)
    parent = m.group(2) or ''
    fields = []
    methods = []
    doc = ''
    i = start + 1
    while i < len(lines):
        cline = lines[i]
        cstripped = cline.lstrip()
        cindent = len(cline) - len(cstripped)
        if cindent == 0 and cstripped:
            break
        if not cstripped:
            i += 1
            continue
        if cstripped.startswith('//') and (not doc):
            doc = cstripped[2:].strip()
        elif cstripped.startswith('fn '):
            fn_m = re.match('fn\\s+(\\w+)\\s*\\(([^)]*)\\)', cstripped)
            if fn_m:
                methods.append({'name': fn_m.group(1), 'params': _parse_params(fn_m.group(2))})
        elif re.match('\\w+\\s*$', cstripped) or re.match('\\w+\\s*=\\s*.+$', cstripped):
            field_m = re.match('(\\w+)(?:\\s*=\\s*(.+))?', cstripped)
            if field_m:
                fields.append({'name': field_m.group(1), 'default': field_m.group(2) or None})
        i += 1
    return {'name': name, 'parent': parent, 'fields': fields, 'methods': methods, 'doc': doc, 'line': start + 1}

def _parse_type(lines: list[str], start: int) -> Optional[dict]:
    line = lines[start].lstrip()
    m = re.match('type\\s+(\\w+)', line)
    if not m:
        return None
    name = m.group(1)
    fields = []
    doc = ''
    i = start + 1
    while i < len(lines):
        cline = lines[i]
        cstripped = cline.lstrip()
        cindent = len(cline) - len(cstripped)
        if cindent == 0 and cstripped:
            break
        if not cstripped:
            i += 1
            continue
        if cstripped.startswith('//') and (not doc):
            doc = cstripped[2:].strip()
        else:
            field_m = re.match('\\s+(\\w+)', cline)
            if field_m:
                fields.append(field_m.group(1))
        i += 1
    return {'name': name, 'fields': fields, 'doc': doc, 'line': start + 1}

def _parse_const(line: str) -> Optional[dict]:
    m = re.match('const\\s+(\\w+)\\s*=\\s*(.+)', line)
    if not m:
        return None
    return {'name': m.group(1), 'value': m.group(2).strip()}

def _parse_params(raw: str) -> list[dict]:
    if not raw.strip():
        return []
    params = []
    for p in raw.split(','):
        p = p.strip()
        if not p:
            continue
        if p.startswith('**'):
            params.append({'name': p[2:], 'kind': 'kwargs'})
        elif p.startswith('*'):
            params.append({'name': p[1:], 'kind': 'args'})
        elif '=' in p:
            parts = p.split('=', 1)
            params.append({'name': parts[0].strip(), 'default': parts[1].strip()})
        else:
            params.append({'name': p})
    return params

def to_markdown(doc: dict, filename: str='') -> str:
    parts = []
    title = filename or 'Module'
    parts.append(f'# {title}\n')
    if doc['module_doc']:
        parts.append(f"{doc['module_doc']}\n")
    if doc['imports']:
        parts.append('## Imports\n')
        for imp in doc['imports']:
            parts.append(f'- `{imp}`')
        parts.append('')
    if doc['constants']:
        parts.append('## Constants\n')
        for c in doc['constants']:
            parts.append(f"- **`{c['name']}`** = `{c['value']}`")
        parts.append('')
    if doc['types']:
        parts.append('## Types\n')
        for t in doc['types']:
            parts.append(f"### `{t['name']}`\n")
            if t['doc']:
                parts.append(f"{t['doc']}\n")
            if t['fields']:
                parts.append('Fields: ' + ', '.join((f'`{f}`' for f in t['fields'])))
                parts.append('')
    if doc['functions']:
        parts.append('## Functions\n')
        for fn in doc['functions']:
            param_str = ', '.join((_param_str(p) for p in fn['params']))
            parts.append(f"### `fn {fn['name']}({param_str})`\n")
            if fn['doc']:
                parts.append(f"{fn['doc']}\n")
            parts.append(f"*Defined at line {fn['line']}*\n")
    if doc['classes']:
        parts.append('## Classes\n')
        for cls in doc['classes']:
            parent_str = f" < {cls['parent']}" if cls['parent'] else ''
            parts.append(f"### `class {cls['name']}{parent_str}`\n")
            if cls['doc']:
                parts.append(f"{cls['doc']}\n")
            if cls['fields']:
                parts.append('**Fields:** ' + ', '.join((f"`{f['name']}`" + (f" = `{f['default']}`" if f['default'] else '') for f in cls['fields'])))
                parts.append('')
            if cls['methods']:
                parts.append('**Methods:**\n')
                for m in cls['methods']:
                    mp = ', '.join((_param_str(p) for p in m['params']))
                    parts.append(f"- `fn {m['name']}({mp})`")
                parts.append('')
    return '\n'.join(parts)

def _param_str(p: dict) -> str:
    if p.get('kind') == 'kwargs':
        return f"**{p['name']}"
    if p.get('kind') == 'args':
        return f"*{p['name']}"
    if 'default' in p:
        return f"{p['name']}={p['default']}"
    return p['name']

def to_html(doc: dict, filename: str='') -> str:
    md = to_markdown(doc, filename)
    html_parts = ['<!DOCTYPE html>', "<html><head><meta charset='utf-8'>", f"<title>{filename or 'TriadLang Docs'}</title>", '<style>body{font-family:system-ui;max-width:800px;margin:2em auto;padding:0 1em;}code{background:#f4f4f4;padding:2px 6px;border-radius:3px;font-size:0.9em;}pre{background:#f4f4f4;padding:1em;border-radius:6px;overflow-x:auto;}h1{border-bottom:2px solid #333;padding-bottom:0.3em;}h2{border-bottom:1px solid #999;padding-bottom:0.2em;margin-top:2em;}h3{color:#555;}</style></head><body>']
    in_code = False
    for line in md.split('\n'):
        if line.startswith('```'):
            if in_code:
                html_parts.append('</pre>')
            else:
                html_parts.append('<pre><code>')
            in_code = not in_code
            continue
        if in_code:
            html_parts.append(_escape(line))
            continue
        if line.startswith('# '):
            html_parts.append(f'<h1>{_escape(line[2:])}</h1>')
        elif line.startswith('## '):
            html_parts.append(f'<h2>{_escape(line[3:])}</h2>')
        elif line.startswith('### '):
            html_parts.append(f'<h3>{_escape(line[4:])}</h3>')
        elif line.startswith('- '):
            html_parts.append(f'<li>{_inline_format(line[2:])}</li>')
        elif line.startswith('*') and line.endswith('*'):
            html_parts.append(f"<p><em>{_escape(line.strip('*'))}</em></p>")
        elif line.strip():
            html_parts.append(f'<p>{_inline_format(line)}</p>')
        else:
            html_parts.append('')
    html_parts.append('</body></html>')
    return '\n'.join(html_parts)

def _escape(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def _inline_format(s: str) -> str:
    s = _escape(s)
    s = re.sub('`([^`]+)`', '<code>\\1</code>', s)
    s = re.sub('\\*\\*([^*]+)\\*\\*', '<strong>\\1</strong>', s)
    return s

def cmd_docgen(args):
    import argparse as _ap
    p = _ap.ArgumentParser(prog='triad docgen')
    p.add_argument('path', help='.tri file or directory')
    p.add_argument('--format', '-f', choices=['markdown', 'html'], default='markdown')
    p.add_argument('--output', '-o', help='Output file (default: stdout)')
    p.add_argument('--recursive', '-r', action='store_true', help='Process directory recursively')
    parsed = p.parse_args(args)
    path = parsed.path
    if os.path.isfile(path):
        with open(path) as f:
            source = f.read()
        doc = parse_tri_docs(source)
        if parsed.format == 'html':
            output = to_html(doc, os.path.basename(path))
        else:
            output = to_markdown(doc, os.path.basename(path))
        if parsed.output:
            with open(parsed.output, 'w') as f:
                f.write(output)
            print(f'Docs written to {parsed.output}')
        else:
            print(output)
    elif os.path.isdir(path):
        tri_files = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.tri'):
                    tri_files.append(os.path.join(root, f))
            if not parsed.recursive:
                break
        all_output = []
        for tf in tri_files:
            with open(tf) as f:
                source = f.read()
            doc = parse_tri_docs(source)
            if parsed.format == 'html':
                all_output.append(to_html(doc, os.path.basename(tf)))
            else:
                all_output.append(to_markdown(doc, os.path.basename(tf)))
        output = '\n\n---\n\n'.join(all_output)
        if parsed.output:
            with open(parsed.output, 'w') as f:
                f.write(output)
            print(f'Docs written to {parsed.output} ({len(tri_files)} files)')
        else:
            print(output)
    else:
        print(f'error: path not found: {path}', file=sys.stderr)
        return 1
    return 0