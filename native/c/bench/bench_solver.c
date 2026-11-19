#define _POSIX_C_SOURCE 199309L
#include "triad_rt.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    printf("=== TriadLang Native Solver Benchmark ===\n");
#ifdef USE_FFTW
    printf("FFT backend: FFTW3\n");
#else
    printf("FFT backend: naive DFT\n");
#endif
    printf("\n");

    double nu_arr[] = {2.0, 0.5, 0.1};
    double lam_arr[] = {-0.3, -0.2, -0.1};

    struct { int N; double T; } configs[] = {
        {128, 2.0},
        {256, 2.0},
        {256, 5.0},
        {512, 2.0},
    };

    for (int c = 0; c < 4; c++) {
        int N = configs[c].N;
        double T = configs[c].T;

        TriadSolverC p = {
            .N = N,
            .L = 32.0,
            .dt = 0.005,
            .T = T,
            .hbar = 1.0,
            .m = 1.0,
            .omega = 0.05,
            .Lambda = -0.5,
            .alpha = 0.15,
            .sigma = 1.5,
            .Gamma = 0.05,
            .f_FDT = 0.002,
            .M = 3,
            .nu = nu_arr,
            .lam = lam_arr,
            .mode = 2,
            .seed = 42,
            .V_ext = "harmonic",
        };

        printf("--- N=%d T=%.1f (dt=0.005, %d steps) ---\n", N, T, (int)(T / p.dt));

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        TriadSolverResult r = triad_solve_1d(&p);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

        double norm = 0, peak = 0;
        for (int i = 0; i < N; i++) {
            norm += r.density_final[i] * r.dx;
            if (r.density_final[i] > peak) peak = r.density_final[i];
        }

        printf("  time:   %.4f s\n", elapsed);
        printf("  norm:   %.6f\n", norm);
        printf("  peak:   %.6f\n", peak);
        printf("  P1+P2+P3 active\n\n");

        triad_solver_result_free(&r);
    }

    return 0;
}
