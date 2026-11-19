/*
 * TriadLang Native Runtime — FFTW3 wrapper
 *
 * Provides 1D FFT/IFFT, FFT frequency grid, and batched operations
 * needed by the split-step Fourier solver.
 *
 * Uses FFTW3 when available at compile time; falls back to a naive DFT
 * so the code compiles everywhere.
 */
#include "triad_rt.h"

#ifdef USE_FFTW
#include <fftw3.h>
#include <math.h>
#else
#include <math.h>
#include <stdlib.h>
#include <string.h>
#endif

/* ── Complex helpers ── */

static TriadCplx cmul(TriadCplx a, TriadCplx b) {
    return (TriadCplx){ a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re };
}

static TriadCplx cadd(TriadCplx a, TriadCplx b) {
    return (TriadCplx){ a.re + b.re, a.im + b.im };
}

static TriadCplx cscale(TriadCplx a, double s) {
    return (TriadCplx){ a.re * s, a.im * s };
}

static double cabs2(TriadCplx a) {
    return a.re*a.re + a.im*a.im;
}

/* ── FFT frequency grid ── */

void triad_fftfreq(int32_t N, double dx, double *out) {
    for (int32_t i = 0; i < N; i++) {
        if (i <= N / 2)
            out[i] = (double)i / ((double)N * dx);
        else
            out[i] = (double)(i - N) / ((double)N * dx);
    }
    for (int32_t i = 0; i < N; i++)
        out[i] *= 2.0 * TRIAD_PI;
}

/* ── FFT / IFFT ── */

#ifdef USE_FFTW

void triad_fft_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    fftw_plan p = fftw_plan_dft_1d(N, (fftw_complex*)in, (fftw_complex*)out,
                                    FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
}

void triad_ifft_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    fftw_plan p = fftw_plan_dft_1d(N, (fftw_complex*)in, (fftw_complex*)out,
                                    FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    for (int32_t i = 0; i < N; i++) {
        out[i].re /= N;
        out[i].im /= N;
    }
}

#else

static void naive_dft(int32_t N, const TriadCplx *in, TriadCplx *out, int sign) {
    double inv_N = 1.0 / N;
    for (int32_t k = 0; k < N; k++) {
        TriadCplx sum = {0, 0};
        for (int32_t n = 0; n < N; n++) {
            double angle = sign * 2.0 * TRIAD_PI * k * n * inv_N;
            TriadCplx w = { cos(angle), sin(angle) };
            sum = cadd(sum, cmul(in[n], w));
        }
        out[k] = (sign == 1) ? sum : cscale(sum, inv_N);
    }
}

void triad_fft_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    naive_dft(N, in, out, 1);
}

void triad_ifft_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    naive_dft(N, in, out, -1);
}

#endif

/* ── In-place element-wise multiply in Fourier space ── */

void triad_fft_fn_mul_inplace(int32_t N, TriadCplx *psi, const TriadCplx *factor) {
    for (int32_t i = 0; i < N; i++)
        psi[i] = cmul(psi[i], factor[i]);
}

/* ── In-place exp(i * V * dt) applied to psi ── */

void triad_apply_potential(int64_t n, TriadCplx *psi,
                           const double *V, double dt_over_hbar) {
    for (int64_t i = 0; i < n; i++) {
        double angle = -V[i] * dt_over_hbar;
        TriadCplx phase = { cos(angle), sin(angle) };
        psi[i] = cmul(psi[i], phase);
    }
}

/* ── Density |psi|^2 (64-bit count) ── */

void triad_density_64(int64_t n, const TriadCplx *psi, double *rho) {
    for (int64_t i = 0; i < n; i++)
        rho[i] = cabs2(psi[i]);
}

/* ═══════════════════════════════════════════════════════════════════
   2D FFT / IFFT  (N x N row-major, stride = N)
   ═══════════════════════════════════════════════════════════════════ */

#ifdef USE_FFTW

void triad_fft2_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    fftw_plan p = fftw_plan_dft_2d(N, N, (fftw_complex*)in, (fftw_complex*)out,
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
}

void triad_ifft2_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    fftw_plan p = fftw_plan_dft_2d(N, N, (fftw_complex*)in, (fftw_complex*)out,
                                   FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    int64_t total = (int64_t)N * N;
    for (int64_t i = 0; i < total; i++) {
        out[i].re /= total;
        out[i].im /= total;
    }
}

#else

void triad_fft2_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    int64_t total = (int64_t)N * N;
    double inv = 1.0 / total;
    for (int32_t kx = 0; kx < N; kx++) {
        for (int32_t ky = 0; ky < N; ky++) {
            TriadCplx sum = {0, 0};
            for (int32_t nx = 0; nx < N; nx++) {
                for (int32_t ny = 0; ny < N; ny++) {
                    double angle = 2.0 * TRIAD_PI * ((double)(kx * nx + ky * ny) / N);
                    TriadCplx w = { cos(angle), sin(angle) };
                    sum = cadd(sum, cmul(in[(int64_t)nx * N + ny], w));
                }
            }
            out[(int64_t)kx * N + ky] = sum;
        }
    }
}

void triad_ifft2_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    int64_t total = (int64_t)N * N;
    double inv = 1.0 / total;
    for (int32_t kx = 0; kx < N; kx++) {
        for (int32_t ky = 0; ky < N; ky++) {
            TriadCplx sum = {0, 0};
            for (int32_t nx = 0; nx < N; nx++) {
                for (int32_t ny = 0; ny < N; ny++) {
                    double angle = -2.0 * TRIAD_PI * ((double)(kx * nx + ky * ny) / N);
                    TriadCplx w = { cos(angle), sin(angle) };
                    sum = cadd(sum, cmul(in[(int64_t)nx * N + ny], w));
                }
            }
            out[(int64_t)kx * N + ky] = cscale(sum, inv);
        }
    }
}

#endif

/* ═══════════════════════════════════════════════════════════════════
   3D FFT / IFFT  (N x N x N row-major, stride = N*N)
   ═══════════════════════════════════════════════════════════════════ */

#ifdef USE_FFTW

void triad_fft3_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    fftw_plan p = fftw_plan_dft_3d(N, N, N, (fftw_complex*)in, (fftw_complex*)out,
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
}

void triad_ifft3_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    fftw_plan p = fftw_plan_dft_3d(N, N, N, (fftw_complex*)in, (fftw_complex*)out,
                                   FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    int64_t total = (int64_t)N * N * N;
    for (int64_t i = 0; i < total; i++) {
        out[i].re /= total;
        out[i].im /= total;
    }
}

#else

void triad_fft3_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    int64_t total = (int64_t)N * N * N;
    for (int32_t kx = 0; kx < N; kx++) {
        for (int32_t ky = 0; ky < N; ky++) {
            for (int32_t kz = 0; kz < N; kz++) {
                TriadCplx sum = {0, 0};
                for (int32_t nx = 0; nx < N; nx++) {
                    for (int32_t ny = 0; ny < N; ny++) {
                        for (int32_t nz = 0; nz < N; nz++) {
                            double angle = 2.0 * TRIAD_PI *
                                ((double)(kx*nx + ky*ny + kz*nz) / N);
                            TriadCplx w = { cos(angle), sin(angle) };
                            sum = cadd(sum, cmul(in[(int64_t)nx*N*N + ny*N + nz], w));
                        }
                    }
                }
                out[(int64_t)kx*N*N + ky*N + kz] = sum;
            }
        }
    }
}

void triad_ifft3_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    int64_t total = (int64_t)N * N * N;
    double inv = 1.0 / total;
    for (int32_t kx = 0; kx < N; kx++) {
        for (int32_t ky = 0; ky < N; ky++) {
            for (int32_t kz = 0; kz < N; kz++) {
                TriadCplx sum = {0, 0};
                for (int32_t nx = 0; nx < N; nx++) {
                    for (int32_t ny = 0; ny < N; ny++) {
                        for (int32_t nz = 0; nz < N; nz++) {
                            double angle = -2.0 * TRIAD_PI *
                                ((double)(kx*nx + ky*ny + kz*nz) / N);
                            TriadCplx w = { cos(angle), sin(angle) };
                            sum = cadd(sum, cmul(in[(int64_t)nx*N*N + ny*N + nz], w));
                        }
                    }
                }
                out[(int64_t)kx*N*N + ky*N + kz] = cscale(sum, inv);
            }
        }
    }
}

#endif
