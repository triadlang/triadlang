/*
 * parity_multi_runtime_2d.c — Native fingerprint of TriadMultiRuntime (D=2),
 * compared against scripts/multi_runtime_2d_to_text.py.
 *
 * Two 2D substrates (N=32, L=16) coupled bidirectionally (density mode),
 * P1+P2+P3 all live, f_FDT=0 to keep RNG out of the comparison.
 *
 * Output format mirrors the Python reference line-for-line; floats use
 * triad_py_repr_float; checksum is order-stable in flat C order.
 */
#include "triad_multi_runtime.h"
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

static void _emit_psi_flat(const TriadCplx *psi, int64_t G) {
    int64_t idxs[2 * NUM_SAMPLES];
    int count = 0;
    int64_t head = (int64_t)NUM_SAMPLES < G ? (int64_t)NUM_SAMPLES : G;
    for (int64_t i = 0; i < head; ++i) idxs[count++] = i;
    if (G > (int64_t)NUM_SAMPLES) {
        int64_t mid = G / 2;
        int64_t tail_end = mid + NUM_SAMPLES;
        if (tail_end > G) tail_end = G;
        for (int64_t i = mid; i < tail_end; ++i) idxs[count++] = i;
    }
    char buf[64];
    for (int j = 0; j < count; ++j) {
        int64_t i = idxs[j];
        _fmt(psi[i].re, buf, sizeof buf); fputs(buf, stdout); fputc(' ', stdout);
        _fmt(psi[i].im, buf, sizeof buf); fputs(buf, stdout);
        if (j + 1 < count) fputc(' ', stdout);
    }
}

static void _checksum(const double *a, int64_t n,
                      double *sum, double *sumsq, double *mx, double *mn) {
    double s = 0.0, sq = 0.0;
    double M = -INFINITY, m = INFINITY;
    for (int64_t i = 0; i < n; ++i) {
        double v = a[i];
        s += v; sq += v * v;
        if (v > M) M = v;
        if (v < m) m = v;
    }
    *sum = s; *sumsq = sq; *mx = M; *mn = m;
}

int main(void) {
    const double L = 16.0;
    const int    N = 32;
    const double dt = 0.01;
    const double T_settle = 0.3;
    const double T_couple = 0.5;
    const int    record_every = 4;

    double nu[1]  = { 0.5 };
    double lam[1] = { 0.03 };

    TriadMultiRuntime *rt = triad_mr_new(dt, record_every);

    int idA = triad_mr_add_substrate(rt, "A", /*D*/2,
        N, L, /*hbar*/1.0, /*m*/1.0, /*omega*/1.0,
        /*Lambda*/-0.3, /*alpha*/0.15, /*sigma*/1.5,
        /*Gamma*/0.05, /*f_FDT*/0.0,
        /*M*/1, nu, lam,
        /*mode*/2, /*seed*/11,
        /*V_ext*/"harmonic", /*psi_init*/NULL);
    int idB = triad_mr_add_substrate(rt, "B", /*D*/2,
        N, L, 1.0, 1.0, 1.0,
        -0.3, 0.15, 1.5,
        0.05, 0.0,
        1, nu, lam,
        2, 22,
        "harmonic", NULL);

    /* settle */
    triad_mr_add_segment(rt, 0.0, T_settle, NULL, 0, NULL, 0);

    /* bidirectional density coupling */
    TriadCouplingEdge *edges = malloc(sizeof(TriadCouplingEdge) * 2);
    edges[0] = (TriadCouplingEdge){ .src_id = idA, .dst_id = idB,
                                    .kappa = -0.2,
                                    .mode = TRIAD_COUPLING_DENSITY };
    edges[1] = (TriadCouplingEdge){ .src_id = idB, .dst_id = idA,
                                    .kappa = -0.15,
                                    .mode = TRIAD_COUPLING_DENSITY };
    triad_mr_add_segment(rt, T_settle, T_settle + T_couple, edges, 2, NULL, 0);

    triad_mr_run(rt);

    char buf[64];
    printf("diverged %d\n", rt->diverged);
    _fmt(rt->global_t, buf, sizeof buf);
    printf("global_t %s\n", buf);

    const char *names[2] = { "A", "B" };
    for (int n_i = 0; n_i < 2; ++n_i) {
        TriadSubstrate *s = triad_mr_find(rt, names[n_i]);
        _fmt(s->dx, buf, sizeof buf);
        printf("sub %s D %d N %d dx %s\n", s->name, s->D, s->N, buf);

        double dxD = pow(s->dx, s->D);
        double norm = 0.0;
        for (int64_t i = 0; i < s->grid_size; ++i)
            norm += s->psi[i].re*s->psi[i].re + s->psi[i].im*s->psi[i].im;
        norm *= dxD;
        _fmt(norm, buf, sizeof buf);
        printf("sub %s norm %s\n", s->name, buf);

        printf("sub %s psi ", s->name);
        _emit_psi_flat(s->psi, s->grid_size);
        putchar('\n');

        printf("sub %s n_records %d\n", s->name, s->n_records);

        if (s->n_records > 0) {
            /* Python density_traj after finalisation has shape
             * (*grid_shape, n_records). For 2D: (N, N, n_records).
             * Flat-C ravel iterates fastest along last axis (n_records),
             * then second-to-last (N), then first (N): order is
             *   for i in N: for j in N: for k in n_records: arr[i,j,k]
             * Our C layout density_traj[k*grid_size + (i*N + j)] needs to
             * be re-packed to that walking order to match the Python
             * checksum. We materialise into a transposed buffer first. */
            int64_t G = s->grid_size;
            int nr = s->n_records;
            double *tr = (double *)malloc(sizeof(double) * (size_t)G * (size_t)nr);
            for (int64_t g = 0; g < G; ++g) {
                for (int k = 0; k < nr; ++k) {
                    tr[g * nr + k] = s->density_traj[(int64_t)k * G + g];
                }
            }
            double sum, sumsq, mx, mn;
            _checksum(tr, G * nr, &sum, &sumsq, &mx, &mn);
            free(tr);
            char b1[64], b2[64], b3[64], b4[64];
            _fmt(sum, b1, sizeof b1);
            _fmt(sumsq, b2, sizeof b2);
            _fmt(mx, b3, sizeof b3);
            _fmt(mn, b4, sizeof b4);
            printf("sub %s dens_sum %s dens_sumsq %s dens_max %s dens_min %s\n",
                   s->name, b1, b2, b3, b4);
        } else {
            printf("sub %s dens_sum 0.0 dens_sumsq 0.0 dens_max 0.0 dens_min 0.0\n",
                   s->name);
        }
    }

    triad_mr_free(rt);
    return 0;
}
