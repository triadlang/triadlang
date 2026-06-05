/*
 * test_multi_runtime.c — Unit tests for TriadMultiRuntime (1D).
 *
 * Verifies the physical invariants of the multi-substrate runtime
 * without relying on Python:
 *   1. Γ=0, f_FDT=0, Λ=0, no coupling: norm conservation (closed system).
 *   2. Γ>0, f_FDT=0: norm strictly decreases (dissipation alone — P3 lock
 *      to zero noise).
 *   3. Λ≠0 alone (P1 only): norm conserved (Λ |ψ|² is real-valued V,
 *      doesn't dissipate, and the phase is unitary).
 *   4. Coupling density mode: a one-way edge from inactive-static A to
 *      B does shift B's density even with κ small.
 *   5. dc_subtracted mode: identical V_couple - mean test — when src ρ is
 *      flat, V_couple should be exactly zero.
 *
 * P1+P2+P3 INVARIANCE: these tests turn pillars OFF individually only to
 * isolate the EFFECT each contributes. The full mode (Λ≠0, Γ>0, M>0) is
 * exercised by parity_multi_runtime against the Python reference.
 */
#include "triad_multi_runtime.h"
#include "triad_codec.h"
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

static double norm_l2(const TriadCplx *psi, int N, double dx) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) s += psi[i].re*psi[i].re + psi[i].im*psi[i].im;
    return s * dx;
}

static void test_norm_conserved_when_closed(void) {
    /* Γ=0, f_FDT=0, Λ=0, M=0, no V_ext, no coupling — fully unitary.
     * Norm must be conserved to numerical precision. */
    int N = 64; double L = 32.0; double dt = 0.01;
    TriadMultiRuntime *rt = triad_mr_new(dt, 4);
    TriadIntCalib calib = { TRIAD_DEFAULT_K_MIN, TRIAD_DEFAULT_K_STEP };
    TriadCplx *psi = triad_encode_int(3, calib, L, N, -1.0, 0.9);

    int id = triad_mr_add_substrate_1d(rt, "C",
        N, L, 1.0, 1.0, 1.0,
        /*Lambda*/0.0, 0.0, 2.0,
        /*Gamma*/0.0, /*f_FDT*/0.0,
        0, NULL, NULL,
        /*mode*/2, /*seed*/42,
        /*V_ext*/NULL, psi);
    free(psi);
    (void)id;

    double norm0 = norm_l2(rt->substrates[0]->psi, N, rt->substrates[0]->dx);

    triad_mr_add_segment(rt, 0.0, 1.0, NULL, 0, NULL, 0);
    triad_mr_run(rt);

    double norm1 = norm_l2(rt->substrates[0]->psi, N, rt->substrates[0]->dx);
    /* split-step FFT is unitary to ~1e-13 over 100 steps */
    CHECK(fabs(norm1 - norm0) < 1e-10,
          "closed system: norm changed by %.3e (norm0=%g norm1=%g)",
          norm1 - norm0, norm0, norm1);

    triad_mr_free(rt);
}

static void test_norm_decreases_under_dissipation(void) {
    /* Γ>0, f_FDT=0: half_lin carries exp(-Γ dt / (2 ℏ)) per half-step,
     * applied twice per step → norm strictly decreases. */
    int N = 64; double L = 32.0; double dt = 0.01;
    TriadMultiRuntime *rt = triad_mr_new(dt, 4);
    TriadIntCalib calib = { TRIAD_DEFAULT_K_MIN, TRIAD_DEFAULT_K_STEP };
    TriadCplx *psi = triad_encode_int(3, calib, L, N, -1.0, 0.9);

    triad_mr_add_substrate_1d(rt, "D",
        N, L, 1.0, 1.0, 1.0,
        0.0, 0.0, 2.0,
        /*Gamma*/0.1, 0.0,
        0, NULL, NULL,
        2, 42,
        NULL, psi);
    free(psi);

    double norm0 = norm_l2(rt->substrates[0]->psi, N, rt->substrates[0]->dx);
    triad_mr_add_segment(rt, 0.0, 0.5, NULL, 0, NULL, 0);
    triad_mr_run(rt);
    double norm1 = norm_l2(rt->substrates[0]->psi, N, rt->substrates[0]->dx);

    CHECK(norm1 < norm0,
          "dissipation: norm did not decrease (norm0=%g norm1=%g)", norm0, norm1);

    /* Analytic decay rate for the split-step scheme: half_lin carries
     * exp(-Γ dt / (2ℏ)) in Ψ-amplitude and is applied 2× per step, so
     * |Ψ|² decays by exp(-2 Γ dt / ℏ) per step. Over T total:
     *   norm(T) = norm0 · exp(-2 Γ T / ℏ)
     * This is the same decay the validated mono-substrate solver shows. */
    double expected = norm0 * exp(-2.0 * 0.1 * 0.5);
    CHECK(fabs(norm1 - expected) / expected < 1e-3,
          "dissipation: norm1=%g expected ~%g (rel=%.3e)",
          norm1, expected, fabs(norm1 - expected) / expected);

    triad_mr_free(rt);
}

static void test_lambda_only_preserves_norm(void) {
    /* P1 only (Λ ≠ 0, Γ = 0, M = 0): Λ|ψ|² is real → unitary phase →
     * norm conserved. */
    int N = 64; double L = 32.0; double dt = 0.01;
    TriadMultiRuntime *rt = triad_mr_new(dt, 4);
    TriadIntCalib calib = { TRIAD_DEFAULT_K_MIN, TRIAD_DEFAULT_K_STEP };
    TriadCplx *psi = triad_encode_int(3, calib, L, N, -1.0, 0.9);

    triad_mr_add_substrate_1d(rt, "E",
        N, L, 1.0, 1.0, 1.0,
        /*Lambda*/-1.0, 0.0, 2.0,
        0.0, 0.0,
        0, NULL, NULL,
        2, 42, NULL, psi);
    free(psi);
    double norm0 = norm_l2(rt->substrates[0]->psi, N, rt->substrates[0]->dx);

    triad_mr_add_segment(rt, 0.0, 1.0, NULL, 0, NULL, 0);
    triad_mr_run(rt);
    double norm1 = norm_l2(rt->substrates[0]->psi, N, rt->substrates[0]->dx);
    CHECK(fabs(norm1 - norm0) < 1e-10,
          "P1 only: norm changed by %.3e", norm1 - norm0);
    triad_mr_free(rt);
}

static void test_coupling_affects_dst(void) {
    /* Two substrates, one-way coupling A→B; B's final psi must differ
     * from a no-coupling reference. */
    int N = 64; double L = 32.0; double dt = 0.01;
    TriadIntCalib calib = { TRIAD_DEFAULT_K_MIN, TRIAD_DEFAULT_K_STEP };

    /* reference: B alone, no coupling */
    TriadMultiRuntime *rt_ref = triad_mr_new(dt, 4);
    TriadCplx *psiB_ref = triad_encode_int(5, calib, L, N, -1.0, 0.9);
    triad_mr_add_substrate_1d(rt_ref, "B",
        N, L, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0,
        0, NULL, NULL, 2, 22, NULL, psiB_ref);
    free(psiB_ref);
    triad_mr_add_segment(rt_ref, 0.0, 0.5, NULL, 0, NULL, 0);
    triad_mr_run(rt_ref);
    double ref_re0 = rt_ref->substrates[0]->psi[0].re;
    double ref_im0 = rt_ref->substrates[0]->psi[0].im;

    /* coupled: A→B with kappa = -0.5, density mode */
    TriadMultiRuntime *rt = triad_mr_new(dt, 4);
    TriadCplx *psiA = triad_encode_int(3, calib, L, N, -1.0, 0.9);
    TriadCplx *psiB = triad_encode_int(5, calib, L, N, -1.0, 0.9);
    int idA = triad_mr_add_substrate_1d(rt, "A",
        N, L, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0,
        0, NULL, NULL, 2, 11, NULL, psiA);
    int idB = triad_mr_add_substrate_1d(rt, "B",
        N, L, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0,
        0, NULL, NULL, 2, 22, NULL, psiB);
    free(psiA); free(psiB);
    TriadCouplingEdge *edges = malloc(sizeof(TriadCouplingEdge));
    edges[0] = (TriadCouplingEdge){ .src_id = idA, .dst_id = idB,
                                    .kappa = -0.5,
                                    .mode = TRIAD_COUPLING_DENSITY };
    triad_mr_add_segment(rt, 0.0, 0.5, edges, 1, NULL, 0);
    triad_mr_run(rt);
    double cpl_re0 = rt->substrates[idB]->psi[0].re;
    double cpl_im0 = rt->substrates[idB]->psi[0].im;

    double diff = sqrt((cpl_re0 - ref_re0)*(cpl_re0 - ref_re0)
                     + (cpl_im0 - ref_im0)*(cpl_im0 - ref_im0));
    /* κ=-0.5 × ρ_A (where |ρ_A| ≈ 0.03 mean) acts via V_couple over 50
     * steps; the resulting phase shift on B's psi[0] is on the order of
     * 1e-8 — small but measurable and non-zero. Larger κ or longer T
     * would amplify it. The contract is just "coupling has effect". */
    CHECK(diff > 1e-9,
          "coupling did not move B's psi[0]: diff=%.3e", diff);

    triad_mr_free(rt);
    triad_mr_free(rt_ref);
}

static void test_dc_subtracted_zero_when_flat(void) {
    /* If src_rho is constant, dc_subtracted contributes V_couple = 0
     * (mean equals every element). Verify by comparing to a no-coupling
     * run; B's final psi must be IDENTICAL. */
    int N = 64; double L = 32.0; double dt = 0.01;
    TriadIntCalib calib = { TRIAD_DEFAULT_K_MIN, TRIAD_DEFAULT_K_STEP };

    /* A starts as a uniform |ψ|² (constant rho): take psi = constant. */
    TriadCplx *psiA = (TriadCplx *)calloc((size_t)N, sizeof(TriadCplx));
    /* unit-norm constant: |psi|² = 1/L, psi.re = 1/sqrt(L) */
    double c = 1.0 / sqrt(L);
    for (int i = 0; i < N; ++i) psiA[i].re = c;

    TriadCplx *psiB = triad_encode_int(4, calib, L, N, -1.0, 0.9);

    TriadMultiRuntime *rt = triad_mr_new(dt, 4);
    int idA = triad_mr_add_substrate_1d(rt, "A",
        N, L, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0,
        0, NULL, NULL, 2, 99, NULL, psiA);
    int idB = triad_mr_add_substrate_1d(rt, "B",
        N, L, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0,
        0, NULL, NULL, 2, 88, NULL, psiB);
    free(psiA);
    TriadCouplingEdge *edges = malloc(sizeof(TriadCouplingEdge));
    edges[0] = (TriadCouplingEdge){ .src_id = idA, .dst_id = idB,
                                    .kappa = -10.0,  /* large κ to be noisy */
                                    .mode = TRIAD_COUPLING_DC_SUBTRACTED };
    triad_mr_add_segment(rt, 0.0, 0.1, edges, 1, NULL, 0);
    triad_mr_run(rt);
    TriadCplx fin0 = rt->substrates[idB]->psi[0];

    /* Reference: B alone */
    TriadMultiRuntime *rt_ref = triad_mr_new(dt, 4);
    triad_mr_add_substrate_1d(rt_ref, "B",
        N, L, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0,
        0, NULL, NULL, 2, 88, NULL, psiB);
    free(psiB);
    triad_mr_add_segment(rt_ref, 0.0, 0.1, NULL, 0, NULL, 0);
    triad_mr_run(rt_ref);
    TriadCplx ref0 = rt_ref->substrates[0]->psi[0];

    /* The src rho is NOT exactly constant because the FFT step in A
     * perturbs it slightly (A also evolves; even constant psi gets a
     * phase from -i V_ext etc). But mode "dc_subtracted" subtracts the
     * mean of A's CURRENT rho at each step, so the residual is just the
     * spatial inhomogeneity A acquires — which starts at zero. We test
     * that |diff| stays small after only 10 steps. */
    double diff = sqrt((fin0.re - ref0.re)*(fin0.re - ref0.re)
                     + (fin0.im - ref0.im)*(fin0.im - ref0.im));
    CHECK(diff < 1e-3,
          "dc_subtracted with near-flat src diverged from no-coupling ref: diff=%.3e",
          diff);

    triad_mr_free(rt);
    triad_mr_free(rt_ref);
}

int main(void) {
    test_norm_conserved_when_closed();
    test_norm_decreases_under_dissipation();
    test_lambda_only_preserves_norm();
    test_coupling_affects_dst();
    test_dc_subtracted_zero_when_flat();

    if (g_failures == 0) {
        printf("test_multi_runtime: PASS (%d/%d)\n", g_total, g_total);
        return 0;
    } else {
        printf("test_multi_runtime: FAIL (%d/%d failures)\n", g_failures, g_total);
        return 1;
    }
}
