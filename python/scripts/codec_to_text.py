import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from runtime.codec import IntCalib, FloatCalib, encode_int, decode_int, encode_bool, decode_bool, encode_float, decode_float, DEFAULT_K_MIN, DEFAULT_K_STEP
NUM_SAMPLES = 4

def _f(x: float) -> str:
    return repr(float(x))

def _i(x: int) -> str:
    return str(int(x))

def _emit_psi(psi: np.ndarray) -> str:
    N = len(psi)
    indices = list(range(min(NUM_SAMPLES, N)))
    if N > NUM_SAMPLES:
        mid = N // 2
        indices += list(range(mid, min(mid + NUM_SAMPLES, N)))
    parts = []
    for i in indices:
        parts.append(_f(psi[i].real))
        parts.append(_f(psi[i].imag))
    return ' '.join(parts)

def _norm(psi: np.ndarray, dx: float) -> float:
    return float((np.abs(psi) ** 2).sum() * dx)

def dump_int_fixture(value: int, L: float, N: int, calib: IntCalib, lines: list) -> None:
    psi = encode_int(value, calib, L, N)
    dx = L / N
    lines.append(f'enc_int {_i(value)} {_f(L)} {_i(N)} {_f(calib.k_min)} {_f(calib.k_step)} psi {_emit_psi(psi)}')
    dec = decode_int(psi, dx, calib)
    lines.append(f'dec_int {_i(value)} -> {_i(dec)}')
    lines.append(f'rt_int {_i(value)} -> {_i(dec)}')
    lines.append(f'norm_ok int {_i(value)} {_f(_norm(psi, dx) - 1.0)}')

def dump_bool_fixture(value: bool, L: float, N: int, lines: list) -> None:
    psi = encode_bool(value, L, N)
    dx = L / N
    lines.append(f'enc_bool {_i(int(value))} {_f(L)} {_i(N)} psi {_emit_psi(psi)}')
    dec = decode_bool(psi, dx)
    if dec is True:
        tri = 1
    elif dec is False:
        tri = 0
    else:
        tri = -1
    lines.append(f'dec_bool {_i(int(value))} -> {_i(tri)}')
    lines.append(f'norm_ok bool {_i(int(value))} {_f(_norm(psi, dx) - 1.0)}')

def dump_float_fixture(value: float, L: float, N: int, calib: FloatCalib, lines: list) -> None:
    psi = encode_float(value, calib, L, N)
    dx = L / N
    lines.append(f'enc_float {_f(value)} {_f(L)} {_i(N)} {_f(calib.a)} {_f(calib.b)} psi {_emit_psi(psi)}')
    dec = decode_float(psi, dx, calib)
    lines.append(f'dec_float {_f(value)} -> {_f(dec)}')
    lines.append(f'rt_float {_f(value)} -> {_f(dec)}')
    lines.append(f'norm_ok float {_f(value)} {_f(_norm(psi, dx) - 1.0)}')

def run() -> str:
    out: list[str] = []
    L = 32.0
    N = 128
    int_calib = IntCalib(k_min=DEFAULT_K_MIN, k_step=DEFAULT_K_STEP)
    float_calib = FloatCalib(a=1.0, b=0.0)
    for v in [0, 1, 2, 3, 5, 10, -1, -3]:
        dump_int_fixture(v, L, N, int_calib, out)
    for v in [True, False]:
        dump_bool_fixture(v, L, N, out)
    for v in [0.5, 1.0, 2.0, 2.5, 5.0]:
        dump_float_fixture(v, L, N, float_calib, out)
    L2, N2 = (64.0, 256)
    dump_int_fixture(4, L2, N2, IntCalib(k_min=0.0, k_step=2 * np.pi / L2), out)
    dump_bool_fixture(True, L2, N2, out)
    dump_float_fixture(1.5, L2, N2, FloatCalib(a=1.0, b=0.0), out)
    return '\n'.join(out) + '\n'

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    sys.stdout.write(run())
if __name__ == '__main__':
    main()