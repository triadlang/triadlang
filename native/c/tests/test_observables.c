/*
 * test_observables.c — Paridade observables nativos contra Python.
 *
 * REGRA: P1+P2+P3 sempre ativos. Observables leem do campo real.
 * Tolerancia macroscopica para campo estocastico (PRNGs diferentes).
 */
#include "triad_observables.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int check(const char *label, double got, double expected, double tol) {
    double rel = fabs(expected) > 1e-12
        ? fabs(got - expected) / fabs(expected)
        : fabs(got - expected);
    int ok = rel < tol;
    printf("  %-28s got=%.10f  ref=%.10f  rel=%.2e  %s\n",
           label, got, expected, rel, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

static int check_positive(const char *label, double got) {
    int ok = got > 0;
    printf("  %-28s got=%.10f  %s\n", label, got, ok ? "OK (>0)" : "FAIL (<=0)");
    return ok ? 0 : 1;
}

static int check_range(const char *label, double got, double lo, double hi) {
    int ok = got >= lo && got <= hi;
    printf("  %-28s got=%.10f  range=[%.4f,%.4f]  %s\n",
           label, got, lo, hi, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main(void) {
    int fails = 0;
    double nu[] = {2.0, 0.5, 0.1};
    double lam[] = {-0.3, -0.2, -0.1};

    /* ═══════════════════════════════════════════════════════════════
       Run solver with full P1+P2+P3 to get field
       ═══════════════════════════════════════════════════════════════ */
    printf("=== Observables from 1D full P1+P2+P3  N=128 T=2.0 ===\n\n");

    TriadSolverC cfg = {0};
    cfg.N = 128; cfg.L = 32.0; cfg.dt = 0.005; cfg.T = 2.0;
    cfg.hbar = 1.0; cfg.m = 1.0; cfg.omega = 0.05;
    cfg.Lambda = -0.5; cfg.alpha = 0.15; cfg.sigma = 1.5;
    cfg.Gamma = 0.05; cfg.f_FDT = 0.002;
    cfg.M = 3; cfg.nu = nu; cfg.lam = lam;
    cfg.mode = 2; cfg.seed = 42; cfg.V_ext = "harmonic";
    cfg.D = 1; cfg.bc = "periodic"; cfg.bc_width = 0.15;

    TriadSolverResult r = triad_solve_1d(&cfg);

    /* ── Field-level scalars ── */
    printf("--- Field-level scalars (§8.1) ---\n");
    printf("  [P1] dispersao hbar*k^2/(2m) + alpha*|k|^sigma\n");
    printf("  [P2] memoria OU: M=%d campos y_j, Lambda=%.2f\n", cfg.M, cfg.Lambda);
    printf("  [P3] dissipacao Gamma=%.3f  ruido FDT f=%.4f\n\n", cfg.Gamma, cfg.f_FDT);

    double obs_norm = triad_obs_norm(r.psi_final, cfg.N, r.dx);
    double obs_peak = triad_obs_peak_density(r.psi_final, cfg.N);
    double obs_fwhm = triad_obs_fwhm(r.psi_final, cfg.N, r.dx);
    double obs_ipr  = triad_obs_ipr(r.psi_final, cfg.N, r.dx);
    double obs_pr   = triad_obs_participation_ratio(r.psi_final, cfg.N, r.dx);

    /* Python ref (different PRNG realization): norm=1.229, peak=0.327, fwhm=1.0,
       ipr=0.098, pr=10.197. Macroscopic tolerance for stochastic field.
       norm (integral) is stable ~15%. peak (pointwise max) fluctuates more
       across realizations because it depends on a single grid point. */
    fails += check("norm", obs_norm, 1.229232, 0.15);
    fails += check("peak_density", obs_peak, 0.327014, 0.25);
    fails += check_positive("fwhm (>0)", obs_fwhm);
    fails += check_positive("ipr (>0)", obs_ipr);
    fails += check_positive("participation_ratio (>0)", obs_pr);
    /* IPR and PR are inverses */
    double ipr_x_pr = obs_ipr * obs_pr;
    fails += check("ipr * pr ~ 1", ipr_x_pr, 1.0, 0.01);

    /* ── Spectral diagnostics ── */
    printf("\n--- Spectral diagnostics (§8.3) ---\n");

    double obs_kstar = triad_obs_dominant_wavenumber(r.psi_final, cfg.N, r.dx, 0.0);
    double k_min_real = 2.0 * TRIAD_PI / cfg.L;
    double obs_kstar_nondc = triad_obs_dominant_wavenumber(r.psi_final, cfg.N, r.dx, k_min_real);
    double obs_cryst = triad_obs_crystallinity(r.psi_final, cfg.N, r.dx, 1.0);

    /* Spectral properties vary more across realizations */
    fails += check_range("k* (>=0)", obs_kstar, 0.0, 50.0);
    fails += check_range("k*_nondc (>=0)", obs_kstar_nondc, 0.0, 50.0);
    fails += check_range("crystallinity [0,1]", obs_cryst, 0.0, 1.0);
    /* Python ref: crystallinity=0.406. Both should have structured component. */
    fails += check_range("crystallinity >0.1", obs_cryst, 0.1, 1.0);

    /* ── Verify power_spectrum ── */
    printf("\n--- power_spectrum sanity ---\n");
    double *k_arr = malloc(sizeof(double) * cfg.N);
    double *P_arr = malloc(sizeof(double) * cfg.N);
    triad_obs_power_spectrum(r.psi_final, cfg.N, r.dx, k_arr, P_arr);
    /* Check sorted ascending */
    int sorted = 1;
    for (int32_t i = 1; i < cfg.N; i++) {
        if (k_arr[i] < k_arr[i-1] - 1e-15) { sorted = 0; break; }
    }
    printf("  k sorted ascending:       %s\n", sorted ? "OK" : "FAIL");
    if (!sorted) fails++;
    /* Check Parseval: sum(P) ~ N * sum(|psi|^2) */
    double P_sum = 0;
    for (int32_t i = 0; i < cfg.N; i++) P_sum += P_arr[i];
    double psi2_sum = 0;
    for (int32_t i = 0; i < cfg.N; i++) psi2_sum += r.psi_final[i].re*r.psi_final[i].re + r.psi_final[i].im*r.psi_final[i].im;
    double parseval_ratio = P_sum / (cfg.N * psi2_sum);
    fails += check("Parseval P_sum/(N*|psi|^2)", parseval_ratio, 1.0, 1e-10);
    free(k_arr);
    free(P_arr);

    /* ── Time-aggregated (using synthetic trajectory) ── */
    printf("\n--- Time-aggregated (§8.5) ---\n");
    /* Build a synthetic stabilizing trajectory: noisy early, flat late */
    int64_t n_traj = 100;
    double *traj = malloc(sizeof(double) * n_traj);
    double *t_traj = malloc(sizeof(double) * n_traj);
    for (int64_t i = 0; i < n_traj; i++) {
        t_traj[i] = (double)i * 0.01;
        if (i < 50)
            traj[i] = 1.0 + 0.5 * sin(i * 0.7);  /* noisy early */
        else
            traj[i] = 1.0 + 0.01 * ((i % 3) - 1);  /* nearly flat late */
    }
    double obs_stab = triad_obs_stabilization_score(traj, n_traj);
    double obs_tts = triad_obs_time_to_stabilize(traj, t_traj, n_traj, 0.1);
    printf("  stabilization_score:      %.6f (expect >0.5 for stabilizing)\n", obs_stab);
    if (obs_stab < 0.5) { printf("    FAIL\n"); fails++; }
    else printf("    OK\n");
    printf("  time_to_stabilize:        %.6f (expect <1.0)\n", obs_tts);
    if (obs_tts >= 1.0) { printf("    FAIL\n"); fails++; }
    else printf("    OK\n");
    free(traj);
    free(t_traj);

    triad_solver_result_free(&r);

    printf("\n=== %s (%d failures) ===\n",
           fails ? "SOME TESTS FAILED" : "ALL TESTS PASSED", fails);
    return fails;
}
