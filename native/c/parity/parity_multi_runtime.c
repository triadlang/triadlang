/*
 * parity_multi_runtime.c — Native fingerprint of TriadMultiRuntime,
 * compared against scripts/multi_runtime_to_text.py.
 *
 * The program is the SAME deterministic two-substrate set-up the Python
 * reference builds: N=64 L=32 dt=0.01, Λ=-0.5, α=0.15, σ=1.5, Γ=0.05, f_FDT=0, memory
 * channel ν=0.5/λ=0.05, V_ext=harmonic, mode=full (P1+P2+P3 all live).
 * Couplings: A→B density κ=-0.3, B→A dc_subtracted κ=-0.2.
 *
 * Output format is line-for-line identical to multi_runtime_to_text.py,
 * with floats formatted via triad_py_repr_float. The aggregate comparator
 * (parity_codec_compare.py-style) tolerates 1-2 ULP in float fields and
 * near-zero residuals, exact match on integer fields.
 */
#include "triad_multi_runtime.h"
#include "triad_codec.h"      /* triad_encode_int */
#include "triad_format.h"     /* triad_py_repr_float */
#include "triad_rt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_SAMPLES 4

static void _fmt(double v, char *out, size_t n) {
    triad_py_repr_float(v, out, n);
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
        _fmt(psi[i].re, buf, sizeof buf); fputs(buf, stdout); fputc(' ', stdout);
        _fmt(psi[i].im, buf, sizeof buf); fputs(buf, stdout);
        if (j + 1 < count) fputc(' ', stdout);
    }
}

/* Order-stable checksum identical to scripts/multi_runtime_to_text.py:
 *   iterate flat index 0..n-1 with scalar accumulators. */
static void _checksum(const double *a, int n,
                      double *out_sum, double *out_sumsq,
                      double *out_max, double *out_min) {
    double s = 0.0, sq = 0.0;
    double mx = -INFINITY, mn = INFINITY;
    for (int i = 0; i < n; ++i) {
        double v = a[i];
        s += v; sq += v * v;
        if (v > mx) mx = v;
        if (v < mn) mn = v;
    }
    *out_sum = s; *out_sumsq = sq; *out_max = mx; *out_min = mn;
}

int main(void) {
    const double L = 32.0;
    const int    N = 64;
    const double dt = 0.01;
    const double T_settle = 0.5;
    const double T_couple = 1.0;
    const int    record_every = 4;

    /* memory chain: M=1 with ν=0.5, λ=0.05 */
    double nu[1]  = { 0.5 };
    double lam[1] = { 0.05 };

    TriadMultiRuntime *rt = triad_mr_new(dt, record_every);

    /* encode initial Psi for A and B (matches encode_int defaults) */
    TriadIntCalib calib = { TRIAD_DEFAULT_K_MIN, TRIAD_DEFAULT_K_STEP };
    TriadCplx *psiA = triad_encode_int(3, calib, L, N, -1.0, 0.9);
    TriadCplx *psiB = triad_encode_int(5, calib, L, N, -1.0, 0.9);

    int id_A = triad_mr_add_substrate_1d(rt, "A",
        N, L, /*hbar*/1.0, /*m*/1.0, /*omega*/1.0,
        /*Lambda*/-0.5, /*alpha*/0.15, /*sigma*/1.5,
        /*Gamma*/0.05, /*f_FDT*/0.0,
        /*M*/1, nu, lam,
        /*mode*/2 /* full */, /*seed*/11,
        /*V_ext*/"harmonic", psiA);
    int id_B = triad_mr_add_substrate_1d(rt, "B",
        N, L, 1.0, 1.0, 1.0,
        -0.5, 0.15, 1.5,
        0.05, 0.0,
        1, nu, lam,
        2, 22,
        "harmonic", psiB);
    free(psiA); free(psiB);
    (void)id_A; (void)id_B;

    /* segment 1: settle (no edges, all active) */
    triad_mr_add_segment(rt, 0.0, T_settle,
                         /*edges*/NULL, /*n_edges*/0,
                         /*active_ids*/NULL, /*n_active*/0);

    /* segment 2: bidirectional coupling */
    TriadCouplingEdge *edges = (TriadCouplingEdge *)malloc(
        sizeof(TriadCouplingEdge) * 2);
    edges[0] = (TriadCouplingEdge){ .src_id = id_A, .dst_id = id_B,
                                    .kappa = -0.3,
                                    .mode = TRIAD_COUPLING_DENSITY };
    edges[1] = (TriadCouplingEdge){ .src_id = id_B, .dst_id = id_A,
                                    .kappa = -0.2,
                                    .mode = TRIAD_COUPLING_DC_SUBTRACTED };
    triad_mr_add_segment(rt, T_settle, T_settle + T_couple,
                         edges, 2,
                         NULL, 0);

    triad_mr_run(rt);

    char buf[64];

    printf("diverged %d\n", rt->diverged);
    _fmt(rt->global_t, buf, sizeof buf);
    printf("global_t %s\n", buf);

    const char *names[2] = { "A", "B" };
    for (int n_i = 0; n_i < 2; ++n_i) {
        TriadSubstrate *s = triad_mr_find(rt, names[n_i]);
        _fmt(s->dx, buf, sizeof buf);
        printf("sub %s N %d dx %s\n", s->name, s->N, buf);

        /* norm */
        double norm = 0.0;
        for (int i = 0; i < s->N; ++i) {
            norm += s->psi[i].re * s->psi[i].re + s->psi[i].im * s->psi[i].im;
        }
        norm *= s->dx;
        _fmt(norm, buf, sizeof buf);
        printf("sub %s norm %s\n", s->name, buf);

        printf("sub %s psi ", s->name);
        _emit_psi(s->psi, s->N);
        putchar('\n');

        printf("sub %s n_records %d\n", s->name, s->n_records);

        if (s->n_records > 0) {
            /* density_traj layout in C: [k * N + i] for time-step k.
             * In Python the saved attribute is shape (N, n_records) — its
             * .T equivalent in row order. The checksum is over the FULL
             * array in flat order; the order has to match Python's
             * (axis=0 fastest in .T means iterating substrate-grid index
             * outermost, time index innermost in the C row-major view).
             * Python does np.array(_rec_buf).T then iterates flat (i*nr+k).
             * Our layout is (n_records, N) flat as (k*N+i). To match
             * Python's iteration order, we have to checksum in the
             * (i*nr + k) walk: outer i, inner k. */
            int nr = s->n_records;
            double *trans = (double *)malloc(sizeof(double) * (size_t)s->N * (size_t)nr);
            for (int i = 0; i < s->N; ++i) {
                for (int k = 0; k < nr; ++k) {
                    trans[i * nr + k] = s->density_traj[k * s->N + i];
                }
            }
            double sum, sumsq, mx, mn;
            _checksum(trans, s->N * nr, &sum, &sumsq, &mx, &mn);
            char b1[64], b2[64], b3[64], b4[64];
            _fmt(sum, b1, sizeof b1);
            _fmt(sumsq, b2, sizeof b2);
            _fmt(mx, b3, sizeof b3);
            _fmt(mn, b4, sizeof b4);
            printf("sub %s dens_sum %s dens_sumsq %s dens_max %s dens_min %s\n",
                   s->name, b1, b2, b3, b4);
            free(trans);
        } else {
            printf("sub %s dens_sum 0.0 dens_sumsq 0.0 dens_max 0.0 dens_min 0.0\n",
                   s->name);
        }
    }

    triad_mr_free(rt);
    return 0;
}
