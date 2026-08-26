
from __future__ import annotations

import importlib

from triad import ntri as np

_T = importlib.import_module('runtime.ml.tensor')

TILE = 128

def _xp_of(a):
    return _T._xp_of(a)

def _acc_dtype(xp, dtype):
    return xp.float64 if dtype == xp.float64 else xp.float32

class _FlashCausalAttention(_T.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        xp = _xp_of(q)
        B, H, T, hd = q.shape
        N = B * H
        in_dtype = q.dtype
        acc = _acc_dtype(xp, in_dtype)

        mm_dt = acc
        scale = float(hd) ** -0.5
        qf = (q.reshape(N, T, hd) * np.dtype(mm_dt).type(scale)
              if mm_dt == xp.float16
              else q.reshape(N, T, hd).astype(acc, copy=False) * scale)
        kf = k.reshape(N, T, hd).astype(mm_dt, copy=False)
        vf = v.reshape(N, T, hd).astype(mm_dt, copy=False)

        out = xp.empty((N, T, hd), dtype=acc)
        lse = xp.empty((N, T), dtype=acc)

        for qs in range(0, T, TILE):
            qe = min(qs + TILE, T)
            qb = qf[:, qs:qe]

            s = (qb @ kf[:, :qe].swapaxes(-1, -2)).astype(acc, copy=False)
            row = xp.arange(qs, qe)[:, None]
            col = xp.arange(qe)[None, :]
            s = s + xp.where(col > row, -xp.inf, 0.0).astype(acc)
            m = s.max(axis=-1)
            p = xp.exp(s - m[..., None])
            den = p.sum(axis=-1)
            pv = p.astype(mm_dt, copy=False) @ vf[:, :qe]
            out[:, qs:qe] = pv.astype(acc, copy=False) / den[..., None]
            lse[:, qs:qe] = m + xp.log(den)

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.in_dtype = in_dtype
        ctx.shape = (B, H, T, hd)
        return out.reshape(B, H, T, hd).astype(in_dtype, copy=False)

    @staticmethod
    def backward(ctx, g):
        q, k, v, out, lse = ctx.saved
        B, H, T, hd = ctx.shape
        xp = _xp_of(q)
        N = B * H
        in_dtype = ctx.in_dtype
        acc = _acc_dtype(xp, in_dtype)
        mm_dt = acc
        scale = float(hd) ** -0.5
        qf = q.reshape(N, T, hd).astype(mm_dt, copy=False)
        kf = k.reshape(N, T, hd).astype(mm_dt, copy=False)
        vf = v.reshape(N, T, hd).astype(mm_dt, copy=False)
        go = g.reshape(N, T, hd).astype(mm_dt, copy=False)

        D = (go.astype(acc, copy=False) * out).sum(axis=-1)

        qf_s = qf * np.dtype(mm_dt).type(scale)
        dq = xp.zeros((N, T, hd), dtype=acc)
        dk = xp.empty((N, T, hd), dtype=acc)
        dv = xp.empty((N, T, hd), dtype=acc)

        for ks in range(0, T, TILE):
            ke = min(ks + TILE, T)
            kb = kf[:, ks:ke]
            vb = vf[:, ks:ke]

            qstrip = qf_s[:, ks:]
            s = (qstrip @ kb.swapaxes(-1, -2)).astype(acc, copy=False)
            row = xp.arange(ks, T)[:, None]
            col = xp.arange(ks, ke)[None, :]
            s = s + xp.where(col > row, -xp.inf, 0.0).astype(acc)
            p = xp.exp(s - lse[:, ks:, None])
            gob = go[:, ks:]
            p_mm = p.astype(mm_dt, copy=False)
            dv[:, ks:ke] = (p_mm.swapaxes(-1, -2) @ gob).astype(acc, copy=False)
            dp = (gob @ vb.swapaxes(-1, -2)).astype(acc, copy=False)
            ds = p * (dp - D[:, ks:, None])
            ds_mm = ds.astype(mm_dt, copy=False)
            dq[:, ks:] += (ds_mm @ kb).astype(acc, copy=False) * scale
            dk[:, ks:ke] = (ds_mm.swapaxes(-1, -2) @ qstrip).astype(acc, copy=False)

        return (dq.reshape(B, H, T, hd).astype(in_dtype, copy=False),
                dk.reshape(B, H, T, hd).astype(in_dtype, copy=False),
                dv.reshape(B, H, T, hd).astype(in_dtype, copy=False))

def flash_causal_attention(q, k, v):

    return _FlashCausalAttention.apply(q, k, v)
