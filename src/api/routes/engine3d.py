from __future__ import annotations

import base64
import secrets
from collections import OrderedDict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from triad import ntri as np

router = APIRouter()

_MAX_SESSIONS = 8
_sessions: OrderedDict[str, object] = OrderedDict()

def _get(sid: str):
    eng = _sessions.get(sid)
    if eng is None:
        raise HTTPException(404, f'unknown session: {sid}')
    _sessions.move_to_end(sid)
    return eng

class CreateReq(BaseModel):
    N: int = Field(32, ge=8, le=64)
    L: float = Field(16.0, gt=0)
    regime: str = 'B0'
    dt: float = Field(0.005, gt=0)
    seed: int = 0

class SpawnReq(BaseModel):
    x: float; y: float; z: float
    width: float = 1.2
    amp: float = 1.0
    vx: float = 0.0; vy: float = 0.0; vz: float = 0.0

class ShapeReq(BaseModel):
    points: list[list[float]]
    width: float = 1.0
    amp: float = 1.0

class WallReq(BaseModel):
    center: list[float]
    size: list[float]
    height: float = 8.0
    gap: float = 0.0

class WellReq(BaseModel):
    center: list[float]
    radius: float
    depth: float = 3.0

class StepReq(BaseModel):
    dt: float = Field(0.1, gt=0, le=2.0)

def _b64_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode()

def _state(eng) -> dict:
    atoms = [{'x': a.x, 'y': a.y, 'z': a.z, 'mass': a.mass, 'peak': a.peak}
             for a in eng.atoms()]
    return {
        't': eng.t,
        'total_mass': eng.total_mass(),
        'atom_count': len(atoms),
        'atoms': atoms,
    }

@router.post('/create')
def create(req: CreateReq):
    from stdlib.engine3d import Engine3D
    while len(_sessions) >= _MAX_SESSIONS:
        _sessions.popitem(last=False)
    sid = secrets.token_hex(8)
    _sessions[sid] = Engine3D(N=req.N, L=req.L, regime=req.regime,
                              dt=req.dt, seed=req.seed)
    eng = _sessions[sid]
    return {'session': sid, 'N': eng.N, 'L': eng.L, **_state(eng)}

@router.delete('/{sid}')
def destroy(sid: str):
    _sessions.pop(sid, None)
    return {'ok': True}

@router.post('/{sid}/spawn')
def spawn(sid: str, req: SpawnReq):
    eng = _get(sid)
    eng.spawn_atom(req.x, req.y, req.z, width=req.width, amp=req.amp,
                   vx=req.vx, vy=req.vy, vz=req.vz)
    return _state(eng)

@router.post('/{sid}/shape')
def shape(sid: str, req: ShapeReq):
    eng = _get(sid)
    eng.place_shape(req.points, width=req.width, amp=req.amp)
    return _state(eng)

@router.post('/{sid}/wall')
def wall(sid: str, req: WallReq):
    eng = _get(sid)
    eng.add_wall_box(req.center, req.size, height=req.height, gap=req.gap)
    return {'ok': True}

@router.post('/{sid}/well')
def well(sid: str, req: WellReq):
    eng = _get(sid)
    eng.add_well_sphere(req.center, req.radius, depth=req.depth)
    return {'ok': True}

@router.post('/{sid}/clear_potential')
def clear_potential(sid: str):
    eng = _get(sid)
    eng.clear_potential()
    return {'ok': True}

@router.post('/{sid}/step')
def step(sid: str, req: StepReq):
    eng = _get(sid)
    eng.step(req.dt)
    return _state(eng)

@router.get('/{sid}/state')
def state(sid: str):
    return _state(_get(sid))

@router.get('/{sid}/density')
def density(sid: str):
    eng = _get(sid)
    rho = eng.density()
    pot = eng._sub.V_ext_static
    return {
        'N': eng.N,
        'L': eng.L,
        't': eng.t,
        'density_b64': _b64_f32(rho),
        'potential_b64': _b64_f32(pot),
        'rho_max': float(rho.max()),
    }
