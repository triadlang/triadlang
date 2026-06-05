/*
 * parity_codec.c — Native fingerprint of the codec API, byte-identical to
 * scripts/codec_to_text.py.
 *
 * Format (one line per scalar, space-separated):
 *   enc_int   <value> <L> <N> <k_min> <k_step> psi <re0> <im0> ...
 *   dec_int   <value> -> <decoded>
 *   rt_int    <value> -> <round_trip>
 *   norm_ok   int <value> <norm-1>
 *
 *   enc_bool  <0|1> <L> <N> psi <re0> <im0> ...
 *   dec_bool  <0|1> -> <tri>             (tri ∈ {1, 0, -1})
 *   norm_ok   bool <0|1> <norm-1>
 *
 *   enc_float <value> <L> <N> <a> <b> psi <re0> <im0> ...
 *   dec_float <value> -> <decoded>
 *   rt_float  <value> -> <round_trip>
 *   norm_ok   float <value> <norm-1>
 *
 * All floats use triad_py_repr_float for round-trippable formatting that
 * matches Python's repr() exactly.
 *
 * P1+P2+P3 note: this driver tests only the codec layer. It does not run
 * the solver. The codecs prepare/read Psi without touching the
 * Λ|Ψ|² / V_mem / α(-Δ)^(σ/2) / FDT machinery, so the equation's three
 * pillars remain intact wherever the resulting Psi is fed to the solver.
 */
#include "triad_codec.h"
#include "triad_format.h"
#include "triad_rt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_SAMPLES 4

static void _fmt(double v, char *out, size_t n) {
    triad_py_repr_float(v, out, n);
}

static double _norm(const TriadCplx *psi, int N, double dx) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) {
        s += psi[i].re * psi[i].re + psi[i].im * psi[i].im;
    }
    return s * dx;
}

static void _emit_psi(const TriadCplx *psi, int N) {
    int idxs[2 * NUM_SAMPLES];
    int count = 0;
    int head = NUM_SAMPLES < N ? NUM_SAMPLES : N;
    for (int i = 0; i < head; ++i) idxs[count++] = i;
    if (N > NUM_SAMPLES) {
        int mid = N / 2;
        int tail_end = mid + NUM_SAMPLES;
        if (tail_end > N) tail_end = N;
        for (int i = mid; i < tail_end; ++i) idxs[count++] = i;
    }
    char buf[64];
    for (int j = 0; j < count; ++j) {
        int i = idxs[j];
        _fmt(psi[i].re, buf, sizeof buf);
        fputs(buf, stdout);
        fputc(' ', stdout);
        _fmt(psi[i].im, buf, sizeof buf);
        fputs(buf, stdout);
        if (j + 1 < count) fputc(' ', stdout);
    }
}

static void dump_int_fixture(long long value, double L, int N,
                             TriadIntCalib calib) {
    TriadCplx *psi = triad_encode_int(value, calib, L, N, -1.0, 0.9);
    if (!psi) { fprintf(stderr, "encode_int alloc failed\n"); exit(1); }
    double dx = L / (double)N;
    char L_s[64], k_min_s[64], k_step_s[64], n_s[64];
    _fmt(L, L_s, sizeof L_s);
    _fmt(calib.k_min, k_min_s, sizeof k_min_s);
    _fmt(calib.k_step, k_step_s, sizeof k_step_s);
    _fmt(_norm(psi, N, dx) - 1.0, n_s, sizeof n_s);

    printf("enc_int %lld %s %d %s %s psi ", value, L_s, N, k_min_s, k_step_s);
    _emit_psi(psi, N);
    putchar('\n');

    long long dec = triad_decode_int(psi, N, dx, calib);
    printf("dec_int %lld -> %lld\n", value, dec);
    printf("rt_int %lld -> %lld\n", value, dec);
    printf("norm_ok int %lld %s\n", value, n_s);

    free(psi);
}

static void dump_bool_fixture(int value, double L, int N) {
    TriadCplx *psi = triad_encode_bool(value, L, N);
    if (!psi) { fprintf(stderr, "encode_bool alloc failed\n"); exit(1); }
    double dx = L / (double)N;
    char L_s[64], n_s[64];
    _fmt(L, L_s, sizeof L_s);
    _fmt(_norm(psi, N, dx) - 1.0, n_s, sizeof n_s);

    printf("enc_bool %d %s %d psi ", value, L_s, N);
    _emit_psi(psi, N);
    putchar('\n');

    int tri = triad_decode_bool(psi, N, dx);
    printf("dec_bool %d -> %d\n", value, tri);
    printf("norm_ok bool %d %s\n", value, n_s);

    free(psi);
}

static void dump_float_fixture(double value, double L, int N,
                               TriadFloatCalib calib) {
    TriadCplx *psi = triad_encode_float(value, calib, L, N);
    if (!psi) { fprintf(stderr, "encode_float alloc failed\n"); exit(1); }
    double dx = L / (double)N;
    char L_s[64], a_s[64], b_s[64], v_s[64], n_s[64];
    _fmt(L, L_s, sizeof L_s);
    _fmt(calib.a, a_s, sizeof a_s);
    _fmt(calib.b, b_s, sizeof b_s);
    _fmt(value, v_s, sizeof v_s);
    _fmt(_norm(psi, N, dx) - 1.0, n_s, sizeof n_s);

    printf("enc_float %s %s %d %s %s psi ", v_s, L_s, N, a_s, b_s);
    _emit_psi(psi, N);
    putchar('\n');

    double dec = triad_decode_float(psi, N, dx, calib);
    char d_s[64]; _fmt(dec, d_s, sizeof d_s);
    printf("dec_float %s -> %s\n", v_s, d_s);
    printf("rt_float %s -> %s\n", v_s, d_s);
    printf("norm_ok float %s %s\n", v_s, n_s);

    free(psi);
}

int main(void) {
    double L = 32.0;
    int N = 128;
    TriadIntCalib int_calib = { TRIAD_DEFAULT_K_MIN, TRIAD_DEFAULT_K_STEP };
    TriadFloatCalib float_calib = { 1.0, 0.0 };

    long long int_values[] = { 0, 1, 2, 3, 5, 10, -1, -3 };
    for (size_t i = 0; i < sizeof(int_values)/sizeof(int_values[0]); ++i) {
        dump_int_fixture(int_values[i], L, N, int_calib);
    }

    dump_bool_fixture(1, L, N);
    dump_bool_fixture(0, L, N);

    double float_values[] = { 0.5, 1.0, 2.0, 2.5, 5.0 };
    for (size_t i = 0; i < sizeof(float_values)/sizeof(float_values[0]); ++i) {
        dump_float_fixture(float_values[i], L, N, float_calib);
    }

    /* one larger-grid sanity check */
    double L2 = 64.0; int N2 = 256;
    TriadIntCalib calib2 = { 0.0, 2.0 * TRIAD_PI / L2 };
    dump_int_fixture(4, L2, N2, calib2);
    dump_bool_fixture(1, L2, N2);
    dump_float_fixture(1.5, L2, N2, float_calib);

    return 0;
}
