/*
 * parity_atoms.c — Native fingerprint of atom/cluster observables,
 * compared against scripts/atoms_to_text.py.
 *
 * Builds the SAME synthetic |Ψ|² fields and runs atom_count_nd,
 * atoms_per_region, atom_separation_1d, and atom_centroids_nd. Output
 * is line-for-line identical to the Python reference; centroids are
 * lex-sorted to make ordering independent of labelling-implementation
 * details (scipy.ndimage label-order vs our BFS label-order).
 */
#include "triad_observables_atoms.h"
#include "triad_format.h"
#include "triad_rt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void _fmt(double v, char *out, size_t n) {
    triad_py_repr_float(v, out, n);
}

/* lex compare for (D)-tuples */
static int g_lex_D = 1;
static int _lex_cmp(const void *a, const void *b) {
    const double *x = (const double *)a;
    const double *y = (const double *)b;
    for (int d = 0; d < g_lex_D; ++d) {
        if (x[d] < y[d]) return -1;
        if (x[d] > y[d]) return 1;
    }
    return 0;
}

static void _emit_centroids(const double *cs, int n, int D) {
    if (n == 0) { printf("0"); return; }
    double *buf = (double *)malloc(sizeof(double) * (size_t)n * (size_t)D);
    memcpy(buf, cs, sizeof(double) * (size_t)n * (size_t)D);
    g_lex_D = D;
    qsort(buf, (size_t)n, sizeof(double) * (size_t)D, _lex_cmp);
    printf("%d", n);
    char tmp[64];
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < D; ++d) {
            _fmt(buf[i * D + d], tmp, sizeof tmp);
            printf(" %s", tmp);
        }
    }
    free(buf);
}

static void dump_fixture(const char *name, const TriadCplx *psi,
                         double dx, int D, int N, double thr) {
    printf("fixture %s\n", name);
    char b[64];
    _fmt(dx, b, sizeof b);
    char bt[64]; _fmt(thr, bt, sizeof bt);
    printf("D %d N %d dx %s thr %s\n", D, N, b, bt);

    int c = triad_atom_count_nd(psi, D, N, dx, thr);
    printf("count %d\n", c);

    double apr = triad_atoms_per_region(psi, D, N, dx, thr);
    _fmt(apr, b, sizeof b);
    printf("per_region %s\n", b);

    if (D == 1) {
        double sep = triad_atom_separation_1d(psi, N, dx, thr);
        _fmt(sep, b, sizeof b);
        printf("separation_1d %s\n", b);
    }

    int n;
    double *cs = triad_atom_centroids_nd(psi, D, N, dx, thr, &n);
    printf("centroids ");
    _emit_centroids(cs ? cs : (double *)"", n, D);
    putchar('\n');
    free(cs);
}

/* ── 1D fixtures (match scripts/atoms_to_text.py) ─────────────── */

static TriadCplx *_make_1d_three_gaussians(int *out_N, double *out_dx) {
    int N = 64; double L = 32.0; double dx = L / N;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)N, sizeof(TriadCplx));
    for (int i = 0; i < N; ++i) {
        double x = ((double)i - (double)(N/2)) * dx;
        double w = 0.8;  /* matches scripts/atoms_to_text.py */
        double v = exp(-((x + 10) * (x + 10)) / (2 * w * w))
                 + exp(-(x * x) / (2 * w * w))
                 + exp(-((x - 10) * (x - 10)) / (2 * w * w));
        psi[i].re = v;
    }
    *out_N = N; *out_dx = dx;
    return psi;
}

static TriadCplx *_make_1d_wraps_boundary(int *out_N, double *out_dx) {
    int N = 64; double L = 32.0; double dx = L / N;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)N, sizeof(TriadCplx));
    double w = 0.6;
    for (int i = 0; i < N; ++i) {
        double x = ((double)i - (double)(N/2)) * dx;
        double v = exp(-((x - 15) * (x - 15)) / (2 * w * w))
                 + exp(-((x + 15) * (x + 15)) / (2 * w * w));
        psi[i].re = v;
    }
    *out_N = N; *out_dx = dx;
    return psi;
}

static TriadCplx *_make_1d_empty(int *out_N, double *out_dx) {
    int N = 32; double L = 16.0; double dx = L / N;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)N, sizeof(TriadCplx));
    *out_N = N; *out_dx = dx;
    return psi;
}

/* ── 2D fixtures ──────────────────────────────────────────────── */

static TriadCplx *_make_2d_four_blobs(int *out_N, double *out_dx) {
    int N = 32; double L = 16.0; double dx = L / N;
    int64_t G = (int64_t)N * N;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)G, sizeof(TriadCplx));
    double w = 0.6;
    for (int i = 0; i < N; ++i) {
        double xi = ((double)i - (double)(N/2)) * dx;
        for (int j = 0; j < N; ++j) {
            double xj = ((double)j - (double)(N/2)) * dx;
            double v = exp(-((xi-4)*(xi-4) + (xj-4)*(xj-4))/(2*w*w))
                     + exp(-((xi+4)*(xi+4) + (xj-4)*(xj-4))/(2*w*w))
                     + exp(-((xi-4)*(xi-4) + (xj+4)*(xj+4))/(2*w*w))
                     + exp(-((xi+4)*(xi+4) + (xj+4)*(xj+4))/(2*w*w));
            psi[i*N + j].re = v;
        }
    }
    *out_N = N; *out_dx = dx;
    return psi;
}

static TriadCplx *_make_2d_single_blob(int *out_N, double *out_dx) {
    int N = 32; double L = 16.0; double dx = L / N;
    int64_t G = (int64_t)N * N;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)G, sizeof(TriadCplx));
    double w = 0.8;
    for (int i = 0; i < N; ++i) {
        double xi = ((double)i - (double)(N/2)) * dx;
        for (int j = 0; j < N; ++j) {
            double xj = ((double)j - (double)(N/2)) * dx;
            double v = exp(-(xi*xi + xj*xj) / (2*w*w));
            psi[i*N + j].re = v;
        }
    }
    *out_N = N; *out_dx = dx;
    return psi;
}

/* ── 3D fixtures ──────────────────────────────────────────────── */

static TriadCplx *_make_3d_two_blobs(int *out_N, double *out_dx) {
    int N = 16; double L = 8.0; double dx = L / N;
    int64_t G = (int64_t)N * N * N;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)G, sizeof(TriadCplx));
    double w = 0.5;
    for (int i = 0; i < N; ++i) {
        double xi = ((double)i - (double)(N/2)) * dx;
        for (int j = 0; j < N; ++j) {
            double xj = ((double)j - (double)(N/2)) * dx;
            for (int k = 0; k < N; ++k) {
                double xk = ((double)k - (double)(N/2)) * dx;
                double v = exp(-((xi-2)*(xi-2) + (xj-2)*(xj-2) + (xk-2)*(xk-2))/(2*w*w))
                         + exp(-((xi+2)*(xi+2) + (xj+2)*(xj+2) + (xk+2)*(xk+2))/(2*w*w));
                psi[(int64_t)i*N*N + j*N + k].re = v;
            }
        }
    }
    *out_N = N; *out_dx = dx;
    return psi;
}

static TriadCplx *_make_3d_single_blob(int *out_N, double *out_dx) {
    int N = 16; double L = 8.0; double dx = L / N;
    int64_t G = (int64_t)N * N * N;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)G, sizeof(TriadCplx));
    double w = 0.8;
    for (int i = 0; i < N; ++i) {
        double xi = ((double)i - (double)(N/2)) * dx;
        for (int j = 0; j < N; ++j) {
            double xj = ((double)j - (double)(N/2)) * dx;
            for (int k = 0; k < N; ++k) {
                double xk = ((double)k - (double)(N/2)) * dx;
                double v = exp(-(xi*xi + xj*xj + xk*xk) / (2*w*w));
                psi[(int64_t)i*N*N + j*N + k].re = v;
            }
        }
    }
    *out_N = N; *out_dx = dx;
    return psi;
}

int main(void) {
    int N; double dx;
    TriadCplx *psi;
    double thr = 0.25;

    psi = _make_1d_three_gaussians(&N, &dx);
    dump_fixture("1d_three_gaussians", psi, dx, 1, N, thr); free(psi);

    psi = _make_1d_wraps_boundary(&N, &dx);
    dump_fixture("1d_wraps_boundary", psi, dx, 1, N, thr); free(psi);

    psi = _make_1d_empty(&N, &dx);
    dump_fixture("1d_empty", psi, dx, 1, N, thr); free(psi);

    psi = _make_2d_four_blobs(&N, &dx);
    dump_fixture("2d_four_blobs", psi, dx, 2, N, thr); free(psi);

    psi = _make_2d_single_blob(&N, &dx);
    dump_fixture("2d_single_blob", psi, dx, 2, N, thr); free(psi);

    psi = _make_3d_two_blobs(&N, &dx);
    dump_fixture("3d_two_blobs", psi, dx, 3, N, thr); free(psi);

    psi = _make_3d_single_blob(&N, &dx);
    dump_fixture("3d_single_blob", psi, dx, 3, N, thr); free(psi);

    return 0;
}
