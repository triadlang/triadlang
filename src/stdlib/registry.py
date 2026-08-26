from __future__ import annotations

import json
import os
import shutil
import subprocess

MANIFEST_FILE = 'triad.json'
MODULES_DIR = 'triad_modules'
GLOBAL_REGISTRY = os.path.expanduser('~/.triad/registry')

class PackageManifest:
    __slots__ = ('name', 'version', 'description', 'dependencies', 'source', 'entry', 'author', 'license')

    def __init__(self, name: str, version: str='0.1.0', description: str='', dependencies: dict[str, str] | None=None, source: str='', entry: str='', author: str='', license: str='MIT'):
        self.name = name
        self.version = version
        self.description = description
        self.dependencies = dependencies or {}
        self.source = source
        self.entry = entry
        self.author = author
        self.license = license

    def to_dict(self) -> dict:
        d = {'name': self.name, 'version': self.version, 'description': self.description, 'dependencies': self.dependencies}
        if self.entry:
            d['entry'] = self.entry
        if self.author:
            d['author'] = self.author
        if self.license != 'MIT':
            d['license'] = self.license
        if self.source:
            d['source'] = self.source
        return d

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> PackageManifest:
        return cls(name=d['name'], version=d.get('version', '0.1.0'), description=d.get('description', ''), dependencies=d.get('dependencies', {}), source=d.get('source', ''), entry=d.get('entry', ''), author=d.get('author', ''), license=d.get('license', 'MIT'))

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write('\n')

def load_manifest(project_dir: str='.') -> PackageManifest | None:
    p = os.path.join(project_dir, MANIFEST_FILE)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return PackageManifest.from_dict(json.load(f))

def save_manifest(manifest: PackageManifest, project_dir: str='.'):
    manifest.save(os.path.join(project_dir, MANIFEST_FILE))

def init_project(name: str, project_dir: str='.') -> PackageManifest:
    manifest = PackageManifest(name=name)
    save_manifest(manifest, project_dir)
    modules_dir = os.path.join(project_dir, MODULES_DIR)
    os.makedirs(modules_dir, exist_ok=True)
    gitignore_path = os.path.join(project_dir, '.gitignore')
    gi_lines: list[str] = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path) as f:
            gi_lines = f.read().splitlines()
    if MODULES_DIR not in gi_lines:
        gi_lines.append(MODULES_DIR + '/')
        with open(gitignore_path, 'w') as f:
            f.write('\n'.join(gi_lines) + '\n')
    return manifest

def _source_kind(source: str) -> str:
    if source.startswith('git+') or source.endswith('.git'):
        return 'git'
    if source.startswith('path:'):
        return 'path'
    if source.startswith('triad://'):
        return 'registry'
    if os.path.isdir(source):
        return 'local'
    return 'unknown'

def install_package(source: str, project_dir: str='.', name: str | None=None, registry_dir: str | None=None) -> str:
    modules_dir = os.path.join(project_dir, MODULES_DIR)
    os.makedirs(modules_dir, exist_ok=True)
    kind = _source_kind(source)
    if kind == 'git':
        return _install_git(source, modules_dir, name)
    elif kind == 'path':
        return _install_path(source, modules_dir, name)
    elif kind == 'registry':
        return _install_registry(source, modules_dir, registry_dir=registry_dir)
    elif kind == 'local':
        return _install_path(source, modules_dir, name)
    else:
        raise RuntimeError(f'unknown package source: {source}')

def _install_git(url: str, modules_dir: str, name: str | None=None) -> str:
    url = url.removeprefix('git+')
    if name is None:
        name = url.rstrip('/').split('/')[-1].removesuffix('.git')
    target = os.path.join(modules_dir, name)
    if os.path.exists(target):
        shutil.rmtree(target)
    subprocess.run(['git', 'clone', '--depth', '1', url, target], check=True, capture_output=True)
    _clean_git_meta(target)
    return name

def _install_path(source: str, modules_dir: str, name: str | None=None) -> str:
    src = source.removeprefix('path:')
    src_abs = os.path.abspath(src)
    if not os.path.isdir(src_abs):
        raise RuntimeError(f'path source is not a directory: {src_abs}')
    if name is None:
        m = load_manifest(src_abs)
        if m:
            name = m.name
        else:
            name = os.path.basename(os.path.normpath(src))
    target = os.path.join(modules_dir, name)
    if os.path.islink(target):
        os.unlink(target)
    elif os.path.exists(target):
        shutil.rmtree(target)
    os.symlink(src_abs, target)
    return name

def _install_registry(source: str, modules_dir: str, registry_dir: str | None=None) -> str:
    reg = registry_dir or GLOBAL_REGISTRY
    spec = source.removeprefix('triad://')
    if '@' in spec:
        name, version = spec.rsplit('@', 1)
    else:
        name = spec
        version = None
    reg_dir = os.path.join(reg, name)
    if not os.path.isdir(reg_dir):
        raise RuntimeError(f"package '{name}' not found in registry ({reg})")
    if version:
        src = os.path.join(reg_dir, version)
        if not os.path.isdir(src):
            vers = [d for d in os.listdir(reg_dir) if os.path.isdir(os.path.join(reg_dir, d))]
            raise RuntimeError(f"version '{version}' not found for '{name}'. available: {vers}")
    else:
        vers = sorted([d for d in os.listdir(reg_dir) if os.path.isdir(os.path.join(reg_dir, d))])
        if not vers:
            raise RuntimeError(f"no versions for '{name}' in registry")
        src = os.path.join(reg_dir, vers[-1])
    target = os.path.join(modules_dir, name)
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.copytree(src, target)
    return name

def _clean_git_meta(pkg_dir: str):
    for sub in ['.git', '__pycache__', '__triadcache__']:
        p = os.path.join(pkg_dir, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)

def install_dependencies(project_dir: str='.') -> list[str]:
    manifest = load_manifest(project_dir)
    if manifest is None:
        return []
    installed: list[str] = []
    for dep_name, dep_source in manifest.dependencies.items():
        install_package(dep_source, project_dir, name=dep_name)
        installed.append(dep_name)
    return installed

def publish_package(project_dir: str='.', registry_dir: str | None=None) -> str:
    manifest = load_manifest(project_dir)
    if manifest is None:
        raise RuntimeError(f'no {MANIFEST_FILE} found in {project_dir}')
    reg = registry_dir or GLOBAL_REGISTRY
    pkg_dir = os.path.join(reg, manifest.name, manifest.version)
    os.makedirs(pkg_dir, exist_ok=True)
    for item in os.listdir(project_dir):
        if item in (MODULES_DIR, '.git', '__pycache__', '__triadcache__', MANIFEST_FILE, '.gitignore'):
            continue
        s = os.path.join(project_dir, item)
        d = os.path.join(pkg_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    shutil.copy2(os.path.join(project_dir, MANIFEST_FILE), os.path.join(pkg_dir, MANIFEST_FILE))
    return os.path.join(reg, manifest.name)

def list_installed(project_dir: str='.') -> list[dict]:
    modules_dir = os.path.join(project_dir, MODULES_DIR)
    if not os.path.isdir(modules_dir):
        return []
    pkgs: list[dict] = []
    for name in sorted(os.listdir(modules_dir)):
        pkg_path = os.path.join(modules_dir, name)
        if not os.path.isdir(pkg_path):
            continue
        m = load_manifest(pkg_path)
        pkgs.append({'name': name, 'version': m.version if m else '?', 'description': m.description if m else '', 'path': pkg_path, 'symlink': os.path.islink(pkg_path)})
    return pkgs

def resolve_modules_dir(project_dir: str='.') -> str:
    return os.path.join(project_dir, MODULES_DIR)

LOCKFILE = 'triad.lock'

import hashlib
import time


def generate_lockfile(project_dir: str = '.') -> dict:

    modules_dir = os.path.join(project_dir, MODULES_DIR)
    lock = {
        'version': 1,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'packages': {}
    }
    if not os.path.isdir(modules_dir):
        return lock
    for name in sorted(os.listdir(modules_dir)):
        pkg_path = os.path.join(modules_dir, name)
        if not os.path.isdir(pkg_path):
            continue
        m = load_manifest(pkg_path)
        if not m:
            continue

        h = hashlib.sha256()
        for root, dirs, files in os.walk(pkg_path):
            dirs.sort()
            for fn in sorted(files):
                fp = os.path.join(root, fn)
                if fn.endswith('.pyc') or '__pycache__' in fp:
                    continue
                rel = os.path.relpath(fp, pkg_path)
                h.update(rel.encode())
                with open(fp, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        h.update(chunk)
        lock['packages'][name] = {
            'version': m.version,
            'source': m.source or 'local',
            'integrity': 'sha256-' + h.hexdigest()[:32],
            'dependencies': m.dependencies
        }
    return lock

def save_lockfile(project_dir: str = '.') -> str:

    lock = generate_lockfile(project_dir)
    path = os.path.join(project_dir, LOCKFILE)
    with open(path, 'w') as f:
        json.dump(lock, f, indent=2)
        f.write('\n')
    return path

def load_lockfile(project_dir: str = '.') -> dict | None:

    path = os.path.join(project_dir, LOCKFILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def verify_lockfile(project_dir: str = '.') -> list[str]:

    lock = load_lockfile(project_dir)
    if not lock:
        return ['no lockfile found']
    errors = []
    modules_dir = os.path.join(project_dir, MODULES_DIR)
    for name, info in lock.get('packages', {}).items():
        pkg_path = os.path.join(modules_dir, name)
        if not os.path.isdir(pkg_path):
            errors.append(f'{name}: package directory missing')
            continue
        m = load_manifest(pkg_path)
        if not m:
            errors.append(f'{name}: manifest missing')
            continue
        if m.version != info.get('version'):
            errors.append(f'{name}: version mismatch (lock={info["version"]}, actual={m.version})')
    return errors

import re as _re

_SEMVER_RE = _re.compile(r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?$')

def _parse_semver(v: str) -> tuple:

    m = _SEMVER_RE.match(v.strip())
    if not m:
        return (0, 0, 0, '')
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4) or '')

def semver_satisfies(version: str, constraint: str) -> bool:

    constraint = constraint.strip()
    if constraint == '*' or constraint == 'latest':
        return True

    v = _parse_semver(version)

    if constraint.startswith('^'):
        floor = _parse_semver(constraint[1:])

        if v[0] != floor[0]:
            return False
        if v[1] > floor[1]:
            return True
        if v[1] < floor[1]:
            return False
        return v[2] >= floor[2]

    if constraint.startswith('~'):
        floor = _parse_semver(constraint[1:])
        if v[0] != floor[0] or v[1] != floor[1]:
            return False
        return v[2] >= floor[2]

    if constraint.startswith('>='):
        floor = _parse_semver(constraint[2:])
        return (v[0], v[1], v[2]) >= (floor[0], floor[1], floor[2])

    if constraint.startswith('<='):
        ceil = _parse_semver(constraint[2:])
        return (v[0], v[1], v[2]) <= (ceil[0], ceil[1], ceil[2])

    if constraint.startswith('>'):
        floor = _parse_semver(constraint[1:])
        return (v[0], v[1], v[2]) > (floor[0], floor[1], floor[2])

    if constraint.startswith('<'):
        ceil = _parse_semver(constraint[1:])
        return (v[0], v[1], v[2]) < (ceil[0], ceil[1], ceil[2])

    return version.strip() == constraint

def resolve_version(package_name: str, constraint: str = '*',
                     registry_dir: str | None = None) -> str | None:

    reg = registry_dir or GLOBAL_REGISTRY
    pkg_dir = os.path.join(reg, package_name)
    if not os.path.isdir(pkg_dir):
        return None
    versions = []
    for vdir in os.listdir(pkg_dir):
        vpath = os.path.join(pkg_dir, vdir)
        if os.path.isdir(vpath) and _SEMVER_RE.match(vdir):
            if semver_satisfies(vdir, constraint):
                versions.append((_parse_semver(vdir), vdir))
    if not versions:
        return None

    versions.sort(key=lambda x: x[0], reverse=True)
    return versions[0][1]
