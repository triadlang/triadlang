from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from triad import ntri as np

router = APIRouter(tags=['ml-interop'])

_TIMEOUT = float(os.environ.get('TRIAD_API_TIMEOUT', '120'))
_MAX_NEW_TOKENS = int(os.environ.get('TRIAD_ML_MAX_NEW_TOKENS', '256'))
_MODEL_CACHE: dict[str, object] = {}

class MLInferenceRequest(BaseModel):
    model: str
    input_ids_b64: str | None = None
    text: str | None = None
    max_new_tokens: int = 32

class MLInferenceResponse(BaseModel):
    output_ids_b64: str
    output_text: str | None = None
    engine: str = "pytorch-interop"

def _allowed_models() -> set[str] | None:
    raw = os.environ.get('TRIAD_ML_ALLOWED_MODELS', '').strip()
    if not raw:
        return set()
    return {m.strip() for m in raw.split(',') if m.strip()}

def _load_model_sync(model_id: str):
    allowed = _allowed_models()
    if not allowed:
        raise PermissionError(
            "remote ML inference disabled. set TRIAD_ML_ALLOWED_MODELS to enable."
        )
    if model_id not in allowed:
        raise PermissionError(
            f"model '{model_id}' not in TRIAD_ML_ALLOWED_MODELS"
        )
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    from runtime.ml.model_loader import load_model
    model = load_model(model_id)
    _MODEL_CACHE[model_id] = model
    return model

def _run_inference_sync(req: MLInferenceRequest) -> MLInferenceResponse:
    try:
        from api.serialization import b64_to_ndarray, ndarray_to_b64
        from runtime.ml.forward import EXTERNAL_STACK, forward_transformer
        from runtime.ml.torch_bridge import from_torch
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f'ML interop not available: {e}')
    if not EXTERNAL_STACK:
        raise HTTPException(status_code=500, detail='ML forward module missing EXTERNAL_STACK marker')

    model = req.model
    if isinstance(model, str):
        model = _load_model_sync(model)

    if req.max_new_tokens > _MAX_NEW_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=f'max_new_tokens > {_MAX_NEW_TOKENS} not allowed via API'
        )

    ids = b64_to_ndarray(req.input_ids_b64) if req.input_ids_b64 else None
    if ids is None and req.text:
        ids = np.array([ord(c) for c in req.text], dtype=np.int64)
    logits = forward_transformer(model, ids, max_new_tokens=req.max_new_tokens)
    out_ids = from_torch(logits).argmax(axis=-1).astype(np.int64)
    return MLInferenceResponse(
        output_ids_b64=ndarray_to_b64(out_ids),
        output_text=''.join(chr(int(c)) for c in out_ids[:128]),
        engine='pytorch-interop',
    )

@router.post('/ml/run', summary='Run external Transformer (PyTorch interop)')
async def ml_run(req: MLInferenceRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_inference_sync, req),
            timeout=_TIMEOUT,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail=f'ML inference timed out after {_TIMEOUT}s')
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result

@router.get('/ml/status', summary='ML interop backend status')
async def ml_status():
    try:
        from runtime.ml.forward import EXTERNAL_STACK
        from runtime.ml.torch_bridge import to_torch
        return {
            'engine': 'pytorch-interop',
            'external_stack': EXTERNAL_STACK,
            'torch_bridge_available': True,
        }
    except ImportError:
        return {'engine': 'unavailable', 'external_stack': False}
