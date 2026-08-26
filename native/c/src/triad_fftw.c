#include "triad_rt.h"

#ifdef USE_FFTW
#include <fftw3.h>
#include <math.h>
#else
#include <math.h>
#include <stdlib.h>
#include <string.h>
#endif

static TriadCplx cmul(TriadCplx a, TriadCplx b) {
    return (TriadCplx){ a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re };
}

static TriadCplx __attribute__((unused)) cadd(TriadCplx a, TriadCplx b) {
    return (TriadCplx){ a.re + b.re, a.im + b.im };
}

static TriadCplx __attribute__((unused)) cscale(TriadCplx a, double s) {
    return (TriadCplx){ a.re * s, a.im * s };
}

static double cabs2(TriadCplx a) {
    return a.re*a.re + a.im*a.im;
}

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

#ifdef USE_FFTW

static fftw_plan _p1d_fwd = NULL, _p1d_bwd = NULL;
static int32_t   _p1d_N   = -1;

static fftw_plan _p2d_fwd = NULL, _p2d_bwd = NULL;
static int32_t   _p2d_N   = -1;

static fftw_plan _p3d_fwd = NULL, _p3d_bwd = NULL;
static int32_t   _p3d_N   = -1;

static void _ensure_plan_1d(int32_t N) {
    if (_p1d_N == N) return;
    fftw_complex *a = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *b = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    if (_p1d_fwd) fftw_destroy_plan(_p1d_fwd);
    if (_p1d_bwd) fftw_destroy_plan(_p1d_bwd);
    _p1d_fwd = fftw_plan_dft_1d(N, a, b, FFTW_FORWARD,  FFTW_ESTIMATE);
    _p1d_bwd = fftw_plan_dft_1d(N, a, b, FFTW_BACKWARD, FFTW_ESTIMATE);
    _p1d_N = N;
    fftw_free(a); fftw_free(b);
}

static void _ensure_plan_2d(int32_t N) {
    if (_p2d_N == N) return;
    int64_t n2 = (int64_t)N * N;
    fftw_complex *a = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n2);
    fftw_complex *b = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n2);
    if (_p2d_fwd) fftw_destroy_plan(_p2d_fwd);
    if (_p2d_bwd) fftw_destroy_plan(_p2d_bwd);
    _p2d_fwd = fftw_plan_dft_2d(N, N, a, b, FFTW_FORWARD,  FFTW_ESTIMATE);
    _p2d_bwd = fftw_plan_dft_2d(N, N, a, b, FFTW_BACKWARD, FFTW_ESTIMATE);
    _p2d_N = N;
    fftw_free(a); fftw_free(b);
}

static void _ensure_plan_3d(int32_t N) {
    if (_p3d_N == N) return;
    int64_t n3 = (int64_t)N * N * N;
    fftw_complex *a = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n3);
    fftw_complex *b = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n3);
    if (_p3d_fwd) fftw_destroy_plan(_p3d_fwd);
    if (_p3d_bwd) fftw_destroy_plan(_p3d_bwd);
    _p3d_fwd = fftw_plan_dft_3d(N, N, N, a, b, FFTW_FORWARD,  FFTW_ESTIMATE);
    _p3d_bwd = fftw_plan_dft_3d(N, N, N, a, b, FFTW_BACKWARD, FFTW_ESTIMATE);
    _p3d_N = N;
    fftw_free(a); fftw_free(b);
}

void triad_fft_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    _ensure_plan_1d(N);
    fftw_execute_dft(_p1d_fwd, (fftw_complex*)in, (fftw_complex*)out);
}

void triad_ifft_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    _ensure_plan_1d(N);
    fftw_execute_dft(_p1d_bwd, (fftw_complex*)in, (fftw_complex*)out);
    double inv = 1.0 / N;
    for (int32_t i = 0; i < N; i++) {
        out[i].re *= inv;
        out[i].im *= inv;
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

void triad_fft_fn_mul_inplace(int32_t N, TriadCplx *psi, const TriadCplx *factor) {
    for (int32_t i = 0; i < N; i++)
        psi[i] = cmul(psi[i], factor[i]);
}

void triad_apply_potential(int64_t n, TriadCplx *psi,
                           const double *V, double dt_over_hbar) {
    for (int64_t i = 0; i < n; i++) {
        double angle = -V[i] * dt_over_hbar;
        TriadCplx phase = { cos(angle), sin(angle) };
        psi[i] = cmul(psi[i], phase);
    }
}

void triad_density_64(int64_t n, const TriadCplx *psi, double *rho) {
    for (int64_t i = 0; i < n; i++)
        rho[i] = cabs2(psi[i]);
}

#ifdef USE_FFTW

void triad_fft2_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    _ensure_plan_2d(N);
    fftw_execute_dft(_p2d_fwd, (fftw_complex*)in, (fftw_complex*)out);
}

void triad_ifft2_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    _ensure_plan_2d(N);
    fftw_execute_dft(_p2d_bwd, (fftw_complex*)in, (fftw_complex*)out);
    int64_t total = (int64_t)N * N;
    double inv = 1.0 / (double)total;
    for (int64_t i = 0; i < total; i++) {
        out[i].re *= inv;
        out[i].im *= inv;
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

#ifdef USE_FFTW

void triad_fft3_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    _ensure_plan_3d(N);
    fftw_execute_dft(_p3d_fwd, (fftw_complex*)in, (fftw_complex*)out);
}

void triad_ifft3_fn(int32_t N, const TriadCplx *in, TriadCplx *out) {
    _ensure_plan_3d(N);
    fftw_execute_dft(_p3d_bwd, (fftw_complex*)in, (fftw_complex*)out);
    int64_t total = (int64_t)N * N * N;
    double inv = 1.0 / (double)total;
    for (int64_t i = 0; i < total; i++) {
        out[i].re *= inv;
        out[i].im *= inv;
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
