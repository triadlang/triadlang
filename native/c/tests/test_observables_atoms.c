/*
 * test_observables_atoms.c — Unit tests for atom/cluster observables.
 *
 * Validates structural invariants that the Python parity test doesn't
 * cover by construction:
 *   1. atom_count_nd is 0 on a null field, regardless of D.
 *   2. atom_count_nd is monotonically non-increasing in threshold_frac.
 *   3. atom_count_2d returns 1 for a single centred Gaussian.
 *   4. atomicity_ratio of a field with itself == 1.
 *   5. atom_count_1d treats wrapping clusters as one (periodic) — a
 *      single blob spanning the right and left edges counts as 1.
 *   6. atom_persistence_late returns 0 when the count is constant
 *      across the late window.
 *
 * P1+P2+P3 invariance: these tests use synthetic ψ; no solver is run.
 * The observables read from ρ = |ψ|² and don't touch the equation —
 * they observe the structure ρ already presents, never imposing.
 */
#include "triad_observables_atoms.h"
#include "triad_rt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_total = 0;

#define CHECK(cond, msg, ...) do {                                    \
    g_total++;                                                        \
    if (!(cond)) {                                                    \
        g_failures++;                                                  \
        fprintf(stderr, "FAIL %s:%d: " msg "\n",                       \
                __FILE__, __LINE__, ##__VA_ARGS__);                    \
    }                                                                  \
} while (0)

static void test_null_field(void) {
    int N = 16;
    int64_t sizes[3] = { N, (int64_t)N * N, (int64_t)N * N * N };
    for (int D = 1; D <= 3; ++D) {
        TriadCplx *psi = (TriadCplx *)calloc((size_t)sizes[D-1], sizeof(TriadCplx));
        int c = triad_atom_count_nd(psi, D, N, 0.5, 0.25);
        CHECK(c == 0, "D=%d null field: count = %d (want 0)", D, c);
        double apr = triad_atoms_per_region(psi, D, N, 0.5, 0.25);
        CHECK(apr == 0.0, "D=%d null field: per_region = %g (want 0)", D, apr);
        free(psi);
    }
}

static void test_monotone_in_threshold(void) {
    /* Build a 1D field with 4 peaks of decreasing amplitude. As the
     * threshold rises, the count must be non-increasing. */
    int N = 64; double L = 32.0; double dx = L / N;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)N, sizeof(TriadCplx));
    double amps[4] = { 1.0, 0.7, 0.5, 0.3 };
    double xs[4]   = { -10.0, -3.0, 4.0, 12.0 };
    for (int i = 0; i < N; ++i) {
        double x = ((double)i - (double)(N/2)) * dx;
        double v = 0;
        for (int p = 0; p < 4; ++p) {
            double dxp = x - xs[p];
            v += amps[p] * exp(-dxp * dxp / (2 * 0.6 * 0.6));
        }
        psi[i].re = v;
    }
    int prev = triad_atom_count_nd(psi, 1, N, dx, 0.01);
    double thrs[5] = { 0.05, 0.1, 0.2, 0.4, 0.7 };
    for (int t = 0; t < 5; ++t) {
        int c = triad_atom_count_nd(psi, 1, N, dx, thrs[t]);
        CHECK(c <= prev, "monotone: thr=%g count=%d, prev=%d", thrs[t], c, prev);
        prev = c;
    }
    free(psi);
}

static void test_single_centered_blob_2d(void) {
    int N = 32; double L = 16.0; double dx = L / N;
    int64_t G = (int64_t)N * N;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)G, sizeof(TriadCplx));
    double w = 0.8;
    for (int i = 0; i < N; ++i) {
        double xi = ((double)i - (double)(N/2)) * dx;
        for (int j = 0; j < N; ++j) {
            double xj = ((double)j - (double)(N/2)) * dx;
            psi[i*N + j].re = exp(-(xi*xi + xj*xj) / (2*w*w));
        }
    }
    int c = triad_atom_count_2d(psi, N, dx, 0.25);
    CHECK(c == 1, "2d single blob: count = %d (want 1)", c);
    int nc;
    double *cs = triad_atom_centroids_2d(psi, N, dx, 0.25, &nc);
    CHECK(nc == 1, "2d single blob: centroid count = %d (want 1)", nc);
    if (nc == 1) {
        CHECK(fabs(cs[0]) < 1e-10 && fabs(cs[1]) < 1e-10,
              "2d single blob centroid = (%g, %g) (want (0,0))", cs[0], cs[1]);
    }
    free(cs); free(psi);
}

static void test_atomicity_ratio_self(void) {
    /* atomicity_ratio(ψ, ψ) must equal 1.0 (each side has the same
     * atoms_per_region). */
    int N = 32; double L = 16.0; double dx = L / N;
    int64_t G = (int64_t)N * N;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)G, sizeof(TriadCplx));
    double w = 0.6;
    for (int i = 0; i < N; ++i) {
        double xi = ((double)i - (double)(N/2)) * dx;
        for (int j = 0; j < N; ++j) {
            double xj = ((double)j - (double)(N/2)) * dx;
            psi[i*N + j].re = exp(-((xi-3)*(xi-3) + xj*xj)/(2*w*w))
                            + exp(-((xi+3)*(xi+3) + xj*xj)/(2*w*w));
        }
    }
    double r = triad_atomicity_ratio(psi, psi, 2, N, dx, 0.25);
    CHECK(fabs(r - 1.0) < 1e-12,
          "atomicity_ratio(psi, psi) = %g (want 1)", r);
    free(psi);
}

static void test_1d_wrap_is_one(void) {
    /* Single Gaussian centred at x = L/2 - dx (very edge) wraps to the
     * left side via periodic BC. count_clusters must return 1 because
     * it's periodic. */
    int N = 64; double L = 32.0; double dx = L / N;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)N, sizeof(TriadCplx));
    /* center the blob at the boundary: cx = -L/2 (which equals +L/2 by periodicity) */
    double cx = -L / 2.0;
    double w = 1.0;
    for (int i = 0; i < N; ++i) {
        double x = ((double)i - (double)(N/2)) * dx;
        /* compute periodic distance */
        double d1 = x - cx;
        double d2 = x - (cx + L);
        double d3 = x - (cx - L);
        double dabs = fabs(d1);
        if (fabs(d2) < dabs) dabs = fabs(d2);
        if (fabs(d3) < dabs) dabs = fabs(d3);
        psi[i].re = exp(-dabs * dabs / (2 * w * w));
    }
    int c = triad_atom_count_1d(psi, N, dx, 0.25);
    CHECK(c == 1, "1d wrap-around blob: count = %d (want 1, periodic)", c);
    free(psi);
}

static void test_persistence_constant_count(void) {
    /* density_traj where every column has exactly 2 peaks: persistence
     * must be 0 (variance = 0). */
    int N = 32; int nrec = 8;
    double *traj = (double *)calloc((size_t)N * (size_t)nrec, sizeof(double));
    double dx = 1.0;
    for (int k = 0; k < nrec; ++k) {
        for (int i = 0; i < N; ++i) {
            double x = (double)i;
            traj[(int64_t)k * N + i] =
                exp(-(x - 8.0)*(x - 8.0) / 4.0)
              + exp(-(x - 24.0)*(x - 24.0) / 4.0);
        }
    }
    double var = triad_atom_persistence_late(traj, nrec, N, dx, 0.25);
    CHECK(fabs(var) < 1e-12,
          "persistence(constant count): var = %g (want 0)", var);
    free(traj);
}

int main(void) {
    test_null_field();
    test_monotone_in_threshold();
    test_single_centered_blob_2d();
    test_atomicity_ratio_self();
    test_1d_wrap_is_one();
    test_persistence_constant_count();

    if (g_failures == 0) {
        printf("test_observables_atoms: PASS (%d/%d)\n", g_total, g_total);
        return 0;
    } else {
        printf("test_observables_atoms: FAIL (%d/%d)\n", g_failures, g_total);
        return 1;
    }
}
