/*
 * parity_multi_runtime_2d_modes.c — Native fingerprint of the 2D
 * multi-runtime exercising dc_subtracted and phase_coherent coupling
 * modes, compared against scripts/multi_runtime_2d_modes_to_text.py.
 *
 * Programme schedule mirrors the Python reference:
 *   seg 1 (0.0 → 0.2)  settle, no edges
 *   seg 2 (0.2 → 0.5)  A→B  dc_subtracted   κ=-0.2
 *   seg 3 (0.5 → 0.8)  A→B  phase_coherent  κ=-0.15  k_target=0.4
 *                      B→A  density         κ=-0.10
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
    int64_t idxs[2 * NUM_SAMPLES]; int count = 0;
    int64_t head = (int64_t)NUM_SAMPLES < G ? (int64_t)NUM_SAMPLES : G;
    for (int64_t i = 0; i < head; ++i) idxs[count++] = i;
    if (G > (int64_t)NUM_SAMPLES) {
        int64_t mid = G / 2;
        int64_t tail = mid + NUM_SAMPLES; if (tail > G) tail = G;
        for (int64_t i = mid; i < tail; ++i) idxs[count++] = i;
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
    double s=0, sq=0, M=-INFINITY, m=INFINITY;
    for (int64_t i = 0; i < n; ++i) {
        double v = a[i];
        s += v; sq += v*v;
        if (v > M) M = v;
        if (v < m) m = v;
    }
    *sum=s; *sumsq=sq; *mx=M; *mn=m;
}

int main(void) {
    const double L = 16.0;
    const int    N = 32;
    const double dt = 0.01;
    const int    record_every = 4;
    double nu[1]  = { 0.5 };
    double lam[1] = { 0.03 };

    TriadMultiRuntime *rt = triad_mr_new(dt, record_every);

    int idA = triad_mr_add_substrate(rt, "A", /*D*/2,
        N, L, 1.0, 1.0, 1.0,
        /*Lambda*/-0.3, 0.15, 1.5,
        /*Gamma*/0.05, 0.0,
        1, nu, lam, 2, 11, "harmonic", NULL);
    int idB = triad_mr_add_substrate(rt, "B", /*D*/2,
        N, L, 1.0, 1.0, 1.0,
        -0.3, 0.15, 1.5,
        0.05, 0.0,
        1, nu, lam, 2, 22, "harmonic", NULL);

    triad_mr_add_segment(rt, 0.0, 0.2, NULL, 0, NULL, 0);

    TriadCouplingEdge *e2 = malloc(sizeof(TriadCouplingEdge));
    e2[0] = (TriadCouplingEdge){ .src_id=idA, .dst_id=idB,
                                 .kappa=-0.2,
                                 .mode=TRIAD_COUPLING_DC_SUBTRACTED,
                                 .k_target=0.0 };
    triad_mr_add_segment(rt, 0.2, 0.5, e2, 1, NULL, 0);

    TriadCouplingEdge *e3 = malloc(sizeof(TriadCouplingEdge) * 2);
    e3[0] = (TriadCouplingEdge){ .src_id=idA, .dst_id=idB,
                                 .kappa=-0.15,
                                 .mode=TRIAD_COUPLING_PHASE_COHERENT,
                                 .k_target=0.4 };
    e3[1] = (TriadCouplingEdge){ .src_id=idB, .dst_id=idA,
                                 .kappa=-0.10,
                                 .mode=TRIAD_COUPLING_DENSITY,
                                 .k_target=0.0 };
    triad_mr_add_segment(rt, 0.5, 0.8, e3, 2, NULL, 0);

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
            int64_t G = s->grid_size; int nr = s->n_records;
            double *tr = malloc(sizeof(double) * (size_t)G * (size_t)nr);
            for (int64_t g = 0; g < G; ++g)
                for (int k = 0; k < nr; ++k)
                    tr[g*nr + k] = s->density_traj[(int64_t)k*G + g];
            double sum, sq, mx, mn;
            _checksum(tr, G * nr, &sum, &sq, &mx, &mn);
            free(tr);
            char b1[64], b2[64], b3[64], b4[64];
            _fmt(sum, b1, sizeof b1); _fmt(sq, b2, sizeof b2);
            _fmt(mx, b3, sizeof b3);  _fmt(mn, b4, sizeof b4);
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
