from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
_state: dict = {'chain': None, 'mempool': []}

def _chain():
    if _state['chain'] is None:
        from stdlib.chain import Chain
        _state['chain'] = Chain()
    return _state['chain']

class TxReq(BaseModel):
    tx: dict

class MineReq(BaseModel):
    miner: str

class WalletReq(BaseModel):
    seed: int | None = None

@router.get('/')
def get_chain():
    ch = _chain()
    return {
        'blocks': len(ch.blocks),
        'difficulty': ch.difficulty,
        'head': ch.blocks[-1].hash,
        'mempool': len(_state['mempool']),
        'chain': [{'index': b.index, 'hash': b.hash, 'nonce': b.nonce,
                   'seal': b.seal, 'txs': len(b.txs)} for b in ch.blocks],
    }

@router.get('/difficulty')
def get_difficulty():
    ch = _chain()
    return {'difficulty': ch.difficulty,
            'history': [ch.difficulty_after(i) for i in range(len(ch.blocks))]}

@router.get('/balance/{address}')
def get_balance(address: str):
    ch = _chain()
    return {'address': address, 'balance': ch.balance(address)}

@router.post('/wallet')
def make_wallet(req: WalletReq):

    from stdlib.chain import Wallet
    if req.seed is not None:
        raise HTTPException(400, 'seed-based wallet rejected: use random entropy')
    w = Wallet()
    return {'address': w.address,
            'warning': 'store the secret locally; it is not returned by the server'}

@router.post('/tx')
def submit_tx(req: TxReq):
    from stdlib.chain import verify_tx
    if not verify_tx(req.tx):
        raise HTTPException(400, 'invalid signature')
    _state['mempool'].append(req.tx)
    return {'ok': True, 'mempool': len(_state['mempool'])}

@router.post('/mine')
def mine(req: MineReq):
    ch = _chain()
    txs = _state['mempool']
    _state['mempool'] = []
    try:
        blk = ch.mine(req.miner, txs)
    except (ValueError, RuntimeError) as e:
        _state['mempool'] = txs
        raise HTTPException(400, str(e))
    return {'index': blk.index, 'hash': blk.hash, 'nonce': blk.nonce,
            'seal': blk.seal, 'difficulty': ch.difficulty}

@router.get('/validate')
def validate(deep: bool = False):
    return _chain().validate(deep=deep)
