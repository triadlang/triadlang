/*
 * test_solver_nd.c — Paridade numerica 1D/2D/3D do solver nativo contra Python.
 *
 * REGRA: P1+P2+P3 sempre ativos. Nenhum modo linear/thermal.
 * Full mode: dispersao + memoria OU + dissipacao + ruido FDT.
 *
 * Observaveis fisicos macroscopicos (norm, peak, crystallinity proxies)
 * devem convergir entre Python e C dentro de tolerancia estatistica.
 * Soma de componentes individuais nao e observavel — depende do PRNG.
 */
#include "triad_rt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double cabs2(TriadCplx z) { return z.re*z.re + z.im*z.im; }

static double max_density(const TriadCplx *p, int64_t n) {
    double mx = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = cabs2(p[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static double compute_norm_1d(const TriadCplx *psi, int32_t N, double dx) {
    double s = 0;
    for (int32_t i = 0; i < N; i++) s += cabs2(psi[i]);
    return s * dx;
}

static double compute_norm_2d(const TriadCplx *psi, int32_t N, double dx) {
    int64_t n = (int64_t)N * N;
    double s = 0;
    for (int64_t i = 0; i < n; i++) s += cabs2(psi[i]);
    return s * dx * dx;
}

static double compute_norm_3d(const TriadCplx *psi, int32_t N, double dx) {
    int64_t n = (int64_t)N * N * N;
    double s = 0;
    for (int64_t i = 0; i < n; i++) s += cabs2(psi[i]);
    return s * dx * dx * dx;
}

static double compute_participation_3d(const TriadCplx *psi, int32_t N, double dx) {
    int64_t n3 = (int64_t)N * N * N;
    double dV = dx * dx * dx;
    double norm2 = 0, rho2 = 0;
    for (int64_t i = 0; i < n3; i++) {
        double d = cabs2(psi[i]);
        norm2 += d;
        rho2 += d * d;
    }
    norm2 *= dV;
    rho2 *= dV;
    return (rho2 > 1e-300) ? norm2 * norm2 / rho2 : 0;
}

static int check(const char *label, double got, double expected, double tol) {
    double rel = fabs(expected) > 1e-12 ? fabs(got - expected) / fabs(expected) : fabs(got - expected);
    int ok = rel < tol;
    printf("  %-25s got=%.10f  ref=%.10f  rel=%.2e  %s\n",
           label, got, expected, rel, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main(void) {
    int fails = 0;
    double nu[] = {2.0, 0.5, 0.1};
    double lam[] = {-0.3, -0.2, -0.1};

    /*
     * Python reference (full P1+P2+P3, seed=42):
     *   1D N=32  T=0.1:  norm=0.980801  peak=0.268777
     *   2D N=16  T=0.05: norm=0.997087  peak=0.077974
     *   3D N=8   T=0.02: norm=1.028689  peak=0.014213
     *                    peak_t0=0.014026  part_t0=79.260
     *
     * Observaveis macroscopicos convergem dentro de ~5% mesmo com PRNGs
     * diferentes porque o ruido FDT e Gaussiano com mesma amplitude.
     * A distribuicao estatistica e a mesma; realizacoes diferem.
     */

    /* ── 1D full P1+P2+P3 ── */
    {
        printf("=== 1D full P1+P2+P3  N=32 T=0.1 ===\n");
        TriadSolverC cfg = {0};
        cfg.N = 32; cfg.L = 32.0; cfg.dt = 0.005; cfg.T = 0.1;
        cfg.hbar = 1.0; cfg.m = 1.0; cfg.omega = 0.05;
        cfg.Lambda = -0.5; cfg.alpha = 0.15; cfg.sigma = 1.5;
        cfg.Gamma = 0.05; cfg.f_FDT = 0.002;
        cfg.M = 3; cfg.nu = nu; cfg.lam = lam;
        cfg.mode = 2; cfg.seed = 42; cfg.V_ext = "harmonic";
        cfg.D = 1; cfg.bc = "periodic"; cfg.bc_width = 0.15;

        TriadSolverResult r = triad_solve_1d(&cfg);
        double norm = compute_norm_1d(r.psi_final, cfg.N, r.dx);
        double peak = max_density(r.psi_final, cfg.N);

        fails += check("norm (macroscopic)", norm, 0.980801342589, 0.05);
        fails += check("peak density", peak, 0.268776900064, 0.05);

        printf("  [P1] dispersao/FFT:     hbar*k^2/(2m) + alpha*|k|^sigma\n");
        printf("  [P2] memoria OU:        M=%d campos y_j\n", cfg.M);
        printf("  [P3] dissipacao+ruído:  Gamma=%.3f  f_FDT=%.4f\n", cfg.Gamma, cfg.f_FDT);
        triad_solver_result_free(&r);
    }

    /* ── 2D full P1+P2+P3 ── */
    {
        printf("\n=== 2D full P1+P2+P3  N=16 T=0.05 ===\n");
        TriadSolverC cfg = {0};
        cfg.N = 16; cfg.L = 32.0; cfg.dt = 0.005; cfg.T = 0.05;
        cfg.hbar = 1.0; cfg.m = 1.0; cfg.omega = 0.05;
        cfg.Lambda = -0.5; cfg.alpha = 0.15; cfg.sigma = 1.5;
        cfg.Gamma = 0.05; cfg.f_FDT = 0.002;
        cfg.M = 3; cfg.nu = nu; cfg.lam = lam;
        cfg.mode = 2; cfg.seed = 42; cfg.V_ext = "harmonic";
        cfg.D = 2; cfg.bc = "periodic"; cfg.bc_width = 0.15;

        TriadSolverResult2D r = triad_solve_2d(&cfg);
        int64_t n2 = (int64_t)cfg.N * cfg.N;
        double norm = compute_norm_2d(r.psi_final, cfg.N, r.dx);
        double peak = max_density(r.psi_final, n2);

        fails += check("norm (macroscopic)", norm, 0.997087416945, 0.05);
        fails += check("peak density", peak, 0.077973724026, 0.05);

        printf("  [P1] 2D FFT + dispersao  [P2] OU M=%d  [P3] FDT noise dx^-1\n", cfg.M);
        triad_solver_result_2d_free(&r);
    }

    /* ── 3D full P1+P2+P3 ── */
    {
        printf("\n=== 3D full P1+P2+P3  N=8 T=0.02 ===\n");
        TriadSolverC cfg = {0};
        cfg.N = 8; cfg.L = 32.0; cfg.dt = 0.005; cfg.T = 0.02;
        cfg.hbar = 1.0; cfg.m = 1.0; cfg.omega = 0.05;
        cfg.Lambda = -0.5; cfg.alpha = 0.15; cfg.sigma = 1.5;
        cfg.Gamma = 0.05; cfg.f_FDT = 0.002;
        cfg.M = 3; cfg.nu = nu; cfg.lam = lam;
        cfg.mode = 2; cfg.seed = 42; cfg.V_ext = "harmonic";
        cfg.D = 3; cfg.bc = "periodic"; cfg.bc_width = 0.15;

        TriadSolverResult3D r = triad_solve_3d(&cfg);
        int64_t n3 = (int64_t)cfg.N * cfg.N * cfg.N;
        double norm = compute_norm_3d(r.psi_final, cfg.N, r.dx);
        double peak = max_density(r.psi_final, n3);

        fails += check("norm (macroscopic)", norm, 1.028689111371, 0.05);
        fails += check("peak density", peak, 0.014212962545, 0.05);

        printf("  observables from solver loop:\n");
        fails += check("peak_t[0] (exact)", r.peak_t[0], 0.014026419312, 1e-10);
        fails += check("part_t[0] (exact)", r.participation_t[0], 79.259722267941, 1e-8);

        printf("  [P1] 3D FFT + dispersao  [P2] OU M=%d  [P3] FDT noise dx^-3/2\n", cfg.M);
        triad_solver_result_3d_free(&r);
    }

    /* ── Conservacao fisica: longa evolucao 1D full, norm deve ficar finito ── */
    {
        printf("\n=== 1D full P1+P2+P3  N=128 T=2.0 (stability) ===\n");
        TriadSolverC cfg = {0};
        cfg.N = 128; cfg.L = 32.0; cfg.dt = 0.005; cfg.T = 2.0;
        cfg.hbar = 1.0; cfg.m = 1.0; cfg.omega = 0.05;
        cfg.Lambda = -0.5; cfg.alpha = 0.15; cfg.sigma = 1.5;
        cfg.Gamma = 0.05; cfg.f_FDT = 0.002;
        cfg.M = 3; cfg.nu = nu; cfg.lam = lam;
        cfg.mode = 2; cfg.seed = 42; cfg.V_ext = "harmonic";
        cfg.D = 1; cfg.bc = "periodic"; cfg.bc_width = 0.15;

        TriadSolverResult r = triad_solve_1d(&cfg);
        double norm = compute_norm_1d(r.psi_final, cfg.N, r.dx);
        double peak = max_density(r.psi_final, cfg.N);
        int norm_ok = (norm > 0.5 && norm < 3.0);
        int peak_ok = (peak > 0.01 && peak < 10.0);
        printf("  norm=%.6f  peak=%.6f  %s\n", norm, peak,
               (norm_ok && peak_ok) ? "OK (bounded)" : "FAIL (diverged)");
        if (!norm_ok || !peak_ok) fails++;
        triad_solver_result_free(&r);
    }

    printf("\n=== %s (%d failures) ===\n",
           fails ? "SOME TESTS FAILED" : "ALL TESTS PASSED", fails);
    return fails;
}
