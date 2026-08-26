from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from triad import ntri as np

SEAL_N = 64
SEAL_L = 16.0
SEAL_T = 0.4
SEAL_DT = 0.01
ROUND = 6
MAX_NONCE = 4096
REWARD = 50

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()

_P = 2 ** 255 - 19
_Q = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_I = pow(2, (_P - 1) // 4, _P)

def _h512(m: bytes) -> bytes:
    return hashlib.sha512(m).digest()

def _xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P)
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P != 0:
        x = x * _I % _P
    if x % 2 != 0:
        x = _P - x
    return x

_BY = 4 * pow(5, _P - 2, _P) % _P
_BX = _xrecover(_BY)
_B = (_BX, _BY)

def _edwards_add(p, q):
    x1, y1 = p
    x2, y2 = q
    x3 = (x1 * y2 + x2 * y1) * pow(1 + _D * x1 * x2 * y1 * y2, _P - 2, _P)
    y3 = (y1 * y2 + x1 * x2) * pow(1 - _D * x1 * x2 * y1 * y2, _P - 2, _P)
    return (x3 % _P, y3 % _P)

def _scalarmult(p, e: int):
    if e == 0:
        return (0, 1)
    q = _scalarmult(p, e // 2)
    q = _edwards_add(q, q)
    if e & 1:
        q = _edwards_add(q, p)
    return q

def _encodepoint(p) -> bytes:
    x, y = p
    bits = [(y >> i) & 1 for i in range(255)] + [x & 1]
    return bytes(sum(bits[i * 8 + j] << j for j in range(8)) for i in range(32))

def _decodepoint(s: bytes):
    y = sum(2 ** i * ((s[i // 8] >> (i % 8)) & 1) for i in range(255))
    x = _xrecover(y)
    if x & 1 != (s[31] >> 7) & 1:
        x = _P - x
    return (x, y)

def _decodeint(s: bytes) -> int:
    return int.from_bytes(s, 'little')

def _encodeint(n: int) -> bytes:
    return n.to_bytes(32, 'little')

def ed25519_publickey(sk: bytes) -> bytes:
    h = _h512(sk)
    a = 2 ** 254 + sum(2 ** i * ((h[i // 8] >> (i % 8)) & 1) for i in range(3, 254))
    return _encodepoint(_scalarmult(_B, a))

def ed25519_sign(msg: bytes, sk: bytes, pk: bytes) -> bytes:
    h = _h512(sk)
    a = 2 ** 254 + sum(2 ** i * ((h[i // 8] >> (i % 8)) & 1) for i in range(3, 254))
    r = _decodeint(_h512(h[32:64] + msg)) % _Q
    R = _scalarmult(_B, r)
    s = (r + _decodeint(_h512(_encodepoint(R) + pk + msg)) % _Q * a) % _Q
    return _encodepoint(R) + _encodeint(s)

def ed25519_verify(msg: bytes, sig: bytes, pk: bytes) -> bool:
    if len(sig) != 64 or len(pk) != 32:
        return False
    try:
        R = _decodepoint(sig[:32])
        A = _decodepoint(pk)
    except (ValueError, TypeError):
        return False
    s = _decodeint(sig[32:64])
    if s >= _Q:
        return False
    h = _decodeint(_h512(sig[:32] + pk + msg)) % _Q
    return _scalarmult(_B, s) == _edwards_add(R, _scalarmult(A, h))

class Wallet:

    def __init__(self, sk: bytes | None = None, seed: int | None = None):
        if sk is None:
            if seed is not None:
                sk = hashlib.sha256(f'triad-wallet-{seed}'.encode()).digest()
            else:
                import secrets
                sk = secrets.token_bytes(32)
        self.sk = sk
        self.pk = ed25519_publickey(sk)
        self.address = self.pk.hex()

    def sign_tx(self, to: str, amount: int, seq: int) -> dict:
        payload = {'from': self.address, 'to': to, 'amount': int(amount), 'seq': int(seq)}
        sig = ed25519_sign(_canon(payload), self.sk, self.pk)
        return {**payload, 'sig': sig.hex()}

def verify_tx(tx: dict) -> bool:
    if tx.get('from') == 'coinbase':
        return tx.get('amount') == REWARD and 'to' in tx
    payload = {'from': tx['from'], 'to': tx['to'],
               'amount': int(tx['amount']), 'seq': int(tx['seq'])}
    try:
        return ed25519_verify(_canon(payload), bytes.fromhex(tx['sig']),
                              bytes.fromhex(tx['from']))
    except (KeyError, ValueError):
        return False

def field_seal(header_hash: str, nonce: int) -> dict:
    from runtime.core.solver import TriadParams, integrate
    from runtime.physics.observables import crystallinity, dominant_wavenumber
    seed = int(_sha256(f'{header_hash}:{nonce}'.encode())[:8], 16)
    p = TriadParams(N=SEAL_N, L=SEAL_L, T=SEAL_T, dt=SEAL_DT, D=1,
                    seed=seed, V_ext=None)
    out = integrate(p, auto_halve_dt=False)
    psi = out['psi_final']
    dx = SEAL_L / SEAL_N
    k_min = 2 * np.pi / SEAL_L
    return {
        'crystallinity': round(float(crystallinity(psi, dx)), ROUND),
        'k_star': round(float(dominant_wavenumber(psi, dx, k_min=k_min)), ROUND),
        'seed': seed,
    }

@dataclass
class Block:
    index: int
    timestamp: int
    prev_hash: str
    txs: list = field(default_factory=list)
    nonce: int = 0
    seal: dict = field(default_factory=dict)
    hash: str = ''

    def header_hash(self) -> str:
        return _sha256(_canon({'index': self.index, 'timestamp': self.timestamp,
                               'prev': self.prev_hash, 'txs': self.txs}))

    def compute_hash(self) -> str:
        return _sha256(_canon({'header': self.header_hash(),
                               'nonce': self.nonce, 'seal': self.seal}))

class Chain:

    def __init__(self):
        self.blocks: list[Block] = []
        genesis = Block(index=0, timestamp=0, prev_hash='0' * 64, txs=[])
        genesis.seal = field_seal(genesis.header_hash(), 0)
        genesis.hash = genesis.compute_hash()
        self.blocks.append(genesis)

    def difficulty_after(self, upto: int) -> float:
        seals = [blk.seal['crystallinity'] for blk in self.blocks[1:upto + 1]]
        if not seals:
            return 0.0
        return round(float(np.median(seals)), ROUND)

    @property
    def difficulty(self) -> float:
        return self.difficulty_after(len(self.blocks) - 1)

    def balances(self) -> dict:
        bal: dict[str, int] = {}
        for blk in self.blocks:
            for tx in blk.txs:
                amt = int(tx['amount'])
                if tx['from'] != 'coinbase':
                    bal[tx['from']] = bal.get(tx['from'], 0) - amt
                bal[tx['to']] = bal.get(tx['to'], 0) + amt
        return bal

    def seqs(self) -> dict:
        out: dict[str, int] = {}
        for blk in self.blocks:
            for tx in blk.txs:
                if tx['from'] != 'coinbase':
                    out[tx['from']] = max(out.get(tx['from'], -1), int(tx['seq']))
        return out

    def balance(self, address: str) -> int:
        return self.balances().get(address, 0)

    def _check_txs(self, txs: list, reward_to_required: bool = True) -> str | None:
        coinbase = [t for t in txs if t.get('from') == 'coinbase']
        if reward_to_required and len(coinbase) != 1:
            return 'bloco precisa de exatamente 1 coinbase'
        bal = self.balances()
        seq = self.seqs()
        for tx in txs:
            if not verify_tx(tx):
                return f'invalid signature: {str(tx)[:60]}'
            if tx['from'] == 'coinbase':
                continue
            amt = int(tx['amount'])
            if amt <= 0:
                return 'amount deve ser positivo'
            if int(tx['seq']) != seq.get(tx['from'], -1) + 1:
                return f'seq de replay: esperado {seq.get(tx["from"], -1) + 1}'
            if bal.get(tx['from'], 0) < amt:
                return f'saldo insuficiente: {tx["from"][:12]}'
            bal[tx['from']] = bal.get(tx['from'], 0) - amt
            bal[tx['to']] = bal.get(tx['to'], 0) + amt
            seq[tx['from']] = int(tx['seq'])
        return None

    def mine(self, miner_address: str, txs: list | None = None,
             timestamp: int | None = None) -> Block:
        txs = list(txs or [])
        txs.append({'from': 'coinbase', 'to': miner_address, 'amount': REWARD})
        err = self._check_txs(txs)
        if err:
            raise ValueError(err)
        prev = self.blocks[-1]
        ts = timestamp if timestamp is not None else prev.timestamp + 1
        blk = Block(index=prev.index + 1, timestamp=ts,
                    prev_hash=prev.hash, txs=txs)
        target = self.difficulty
        hh = blk.header_hash()
        for nonce in range(MAX_NONCE):
            seal = field_seal(hh, nonce)
            if seal['crystallinity'] >= target:
                blk.nonce = nonce
                blk.seal = seal
                blk.hash = blk.compute_hash()
                self.blocks.append(blk)
                return blk
        raise RuntimeError(f'did not seal in {MAX_NONCE} attempts (difficulty {target})')

    def validate(self, deep: bool = False) -> dict:
        for i, blk in enumerate(self.blocks):
            if blk.hash != blk.compute_hash():
                return {'ok': False, 'block': i, 'reason': 'hash mismatch'}
            if i == 0:
                continue
            if blk.prev_hash != self.blocks[i - 1].hash:
                return {'ok': False, 'block': i, 'reason': 'elo quebrado'}
            seal = field_seal(blk.header_hash(), blk.nonce)
            if seal != blk.seal:
                return {'ok': False, 'block': i, 'reason': 'physical seal does not reproduce'}
            target = self.difficulty_after(i - 1)
            if blk.seal['crystallinity'] < target:
                return {'ok': False, 'block': i,
                        'reason': f'cristalinidade abaixo da dificuldade {target}'}
            stub = Chain.__new__(Chain)
            stub.blocks = self.blocks[:i]
            err = stub._check_txs(blk.txs)
            if err:
                return {'ok': False, 'block': i, 'reason': err}
            if deep:
                for n in range(blk.nonce):
                    if field_seal(blk.header_hash(), n)['crystallinity'] >= target:
                        return {'ok': False, 'block': i,
                                'reason': f'non-minimal nonce: {n} already sealed'}
        return {'ok': True, 'blocks': len(self.blocks)}

    def to_json(self) -> str:
        return json.dumps([asdict(b) for b in self.blocks], indent=1)

    @classmethod
    def from_json(cls, data: str) -> Chain:
        ch = cls.__new__(cls)
        ch.blocks = [Block(**b) for b in json.loads(data)]
        return ch
