from __future__ import annotations

import hashlib
import json
import os
import time

from runtime.env import REPO_ROOT, env
from runtime.physics.atoms import CrystalMemory, read_lattice
from triad import ntri as np


def memory_dir() -> str:
    d = env('TRIAD_MEMORY_DIR', 'memory')
    return d if os.path.isabs(d) else os.path.join(REPO_ROOT, d)

def _text_vector(text: str, dim: int = 256) -> np.ndarray:
    t = ' ' + text.lower().strip() + ' '
    v = np.zeros(dim)
    for i in range(len(t) - 2):
        g = t[i:i + 3]
        h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
        v[h % dim] += 1.0 if (h >> 8) % 2 else -1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

class TriadMemory:
    def __init__(self, root: str = None, N: int = 64, D: int = 3,
                 seed: int = 42, backend: str = None):
        if root is None:
            root = memory_dir()
        if backend is None:
            backend = env('TRIAD_MEMORY_BACKEND', 'auto')
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.field_path = os.path.join(root, 'field.npz')
        self.book_path = os.path.join(root, 'book.jsonl')
        self.config_path = os.path.join(root, 'config.json')
        if os.path.exists(self.config_path):
            cfg = json.load(open(self.config_path))
            N, D, seed = cfg['N'], cfg['D'], cfg['seed']
        self.cfg = {'N': N, 'D': D, 'seed': seed}
        self.mem = CrystalMemory(N=N, D=D, seed=seed, backend=backend)
        self.n_sites = len(self.mem.sites)
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((256, self.n_sites))
        self._k = max(8, self.n_sites // 8)
        if os.path.exists(self.field_path):
            data = np.load(self.field_path)
            self.mem.psi = data['psi']
            self.mem.y = data['y']

    def _save(self):
        np.savez_compressed(self.field_path,
                            psi=np.asarray(self.mem.psi),
                            y=np.asarray(self.mem.y))
        with open(self.config_path, 'w') as f:
            json.dump(self.cfg, f)

    def _book(self):
        if not os.path.exists(self.book_path):
            return []
        with open(self.book_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def signature(self, text: str) -> list:
        weights = _text_vector(text) @ self._proj
        return sorted(np.argsort(weights)[-self._k:].tolist())

    def start(self, T: float = 10.0):
        self.mem.crystallize(T)
        self._save()
        return {'sites': self.n_sites, 'T': T}

    def record(self, text: str, tags: list = None, settle_T: float = 0.5):
        if not os.path.exists(self.field_path):
            raise RuntimeError('field not initialized — run `start` first')
        sites = self.signature(text)
        for s in sites:
            self.mem.write(int(s), phase=0.0)
        self.mem.settle(settle_T)
        item = {'id': hashlib.md5(
                    (text + str(time.time())).encode()).hexdigest()[:10],
                'text': text, 'sites': sites,
                'tags': tags or [], 'when': time.strftime('%Y-%m-%d %H:%M')}
        with open(self.book_path, 'a') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        self._save()
        return item['id']

    def _occupancy(self) -> np.ndarray:
        info = read_lattice(np.asarray(self.mem.psi), self.mem.dx,
                            self.mem.sites)
        return np.array([s['occupied'] for s in info], dtype=bool)

    def recall(self, query: str, n: int = None) -> list:
        if n is None:
            n = env('TRIAD_MEMORY_RECALL_N', 5)
        book = self._book()
        if not book:
            return []
        occ = self._occupancy()
        q = set(self.signature(query))
        output = []
        for item in book:
            s = set(item['sites'])
            sim = len(q & s) / max(len(q | s), 1)
            viv = float(occ[list(s)].mean()) if s else 0.0
            output.append({'id': item['id'], 'text': item['text'],
                          'when': item['when'], 'sim': round(sim, 3),
                          'liveness': round(viv, 3),
                          'score': round(sim * viv, 4)})
        output.sort(key=lambda x: -x['score'])
        return output[:n]

    def resonate(self, query: str, n: int = None, T: float = None,
                force: float = None) -> list:
        if n is None:
            n = env('TRIAD_MEMORY_RECALL_N', 5)
        if T is None:
            T = env('TRIAD_MEMORY_RESONATE_T', 0.5)
        if force is None:
            force = env('TRIAD_MEMORY_RESONATE_FORCE', 0.25)
        book = self._book()
        if not book:
            return []
        psi0 = np.array(self.mem.psi, copy=True)
        y0 = np.array(self.mem.y, copy=True)
        sites_q = self.signature(query)

        self.mem._evolve(T)
        base = read_lattice(np.asarray(self.mem.psi), self.mem.dx,
                            self.mem.sites)
        m_base = np.array([s['mass'] for s in base])

        self.mem.psi = np.array(psi0, copy=True)
        self.mem.y = np.array(y0, copy=True)
        peak = float(np.abs(self.mem.psi).max())
        for s in sites_q:
            self.mem.write(int(s), phase=0.0, amplitude=force * peak)
        self.mem._evolve(T)
        pert = read_lattice(np.asarray(self.mem.psi), self.mem.dx,
                            self.mem.sites)
        m_pert = np.array([s['mass'] for s in pert])

        self.mem.psi, self.mem.y = psi0, y0

        resp = np.abs(m_pert - m_base)
        scale = float(m_base.mean()) or 1.0
        occ = self._occupancy()
        q = set(sites_q)
        output = []
        for item in book:
            s = item['sites']
            outside = [x for x in s if x not in q]
            reson = float(resp[s].mean()) / scale if s else 0.0
            echo = float(resp[outside].mean()) / scale if outside else 0.0
            viv = float(occ[s].mean()) if s else 0.0
            output.append({'id': item['id'], 'text': item['text'],
                          'when': item['when'],
                          'resonance': round(reson, 4),
                          'echo': round(echo, 4),
                          'liveness': round(viv, 3)})
        output.sort(key=lambda x: -(x['resonance'] * x['liveness']))
        return output[:n]

    def sleep(self, T: float = None):
        if T is None:
            T = env('TRIAD_MEMORY_SLEEP_T', 1.0)
        self.mem.settle(T)
        self._save()
        occ = self._occupancy()
        return {'T': T, 'sites_alive': int(occ.sum()),
                'sites_total': self.n_sites}

    def status(self):
        book = self._book()
        alive = None
        if os.path.exists(self.field_path):
            occ = self._occupancy()
            alive = int(occ.sum())
        return {'items_in_book': len(book),
                'sites_alive': alive, 'sites_total': self.n_sites,
                'field_started': os.path.exists(self.field_path),
                'config': self.cfg}
