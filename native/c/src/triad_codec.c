#include "triad_codec.h"
#include "triad_observables.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline double _grid_dx(double L, int N) {
    return L / (double)N;
}

static inline double _grid_x(int i, double L, int N) {
    return -0.5 * L + (double)i * (L / (double)N);
}

static double _norm_l2(const TriadCplx *psi, int N, double dx) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) {
        s += psi[i].re * psi[i].re + psi[i].im * psi[i].im;
    }
    return s * dx;
}

static void _normalise(TriadCplx *psi, int N, double dx) {
    double n = _norm_l2(psi, N, dx);
    if (n <= 0.0) return;
    double inv = 1.0 / sqrt(n);
    for (int i = 0; i < N; ++i) {
        psi[i].re *= inv;
        psi[i].im *= inv;
    }
}

TriadCplx *triad_encode_int(long long value, TriadIntCalib calib,
                            double L, int N,
                            double envelope_width, double modulation) {
    if (N <= 0) return NULL;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)N, sizeof(TriadCplx));
    if (!psi) return NULL;

    double dx = _grid_dx(L, N);
    double k_n = calib.k_min + (double)value * calib.k_step;

    if (envelope_width <= 0.0) envelope_width = L / 4.0;

    if (k_n == 0.0) {

        for (int i = 0; i < N; ++i) {
            psi[i].re = 1.0;
            psi[i].im = 0.0;
        }
    } else {

        for (int i = 0; i < N; ++i) {
            double x = _grid_x(i, L, N);
            double m = 1.0 + modulation * cos(k_n * x);
            psi[i].re = m;
            psi[i].im = 0.0;
        }
    }

    _normalise(psi, N, dx);
    return psi;
}

long long triad_decode_int(const TriadCplx *psi, int N, double dx,
                           TriadIntCalib calib) {

    if (N < 4) return 0;
    double L = (double)N * dx;

    TriadCplx *psi_hat = (TriadCplx *)malloc(sizeof(TriadCplx) * N);
    triad_fft_fn(N, psi, psi_hat);
    double *P = (double *)malloc(sizeof(double) * N);
    for (int i = 0; i < N; ++i)
        P[i] = psi_hat[i].re * psi_hat[i].re + psi_hat[i].im * psi_hat[i].im;
    free(psi_hat);

    int half_end = N / 2;
    if (half_end <= 1) {
        free(P);
        return 0;
    }

    int vbin = 1;
    double best = P[1];
    for (int i = 2; i < half_end; ++i) {
        if (P[i] > best) {
            best = P[i];
            vbin = i;
        }
    }

    if (P[vbin] < 1e-6 * P[0]) {
        free(P);
        return 0;
    }

    double k_star = (double)vbin * (2.0 * TRIAD_PI / L);
    double nf = (k_star - calib.k_min) / calib.k_step;
    long long n = (long long)llrint(nf);
    free(P);
    return n;
}

TriadCplx *triad_encode_bool(int value, double L, int N) {
    if (N <= 0) return NULL;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)N, sizeof(TriadCplx));
    if (!psi) return NULL;

    double dx = _grid_dx(L, N);

    if (value) {

        double w = L / 6.0;
        double denom = 2.0 * w * w;
        for (int i = 0; i < N; ++i) {
            double x = _grid_x(i, L, N);
            double env = exp(-(x*x) / denom);
            psi[i].re = env * cos(3.0 * x);
            psi[i].im = 0.0;
        }
    } else {

        double w = L / 3.0;
        double denom = 2.0 * w * w;
        for (int i = 0; i < N; ++i) {
            double x = _grid_x(i, L, N);
            psi[i].re = exp(-(x*x) / denom);
            psi[i].im = 0.0;
        }
    }

    _normalise(psi, N, dx);
    return psi;
}

int triad_decode_bool(const TriadCplx *psi, int N, double dx) {
    double c = triad_obs_crystallinity(psi, N, dx, 1.0);
    if (c > 0.5) return TRIAD_BOOL_TRUE;
    if (c < 0.2) return TRIAD_BOOL_FALSE;
    return TRIAD_BOOL_AMBIGUOUS;
}

TriadCplx *triad_encode_float(double value, TriadFloatCalib calib,
                              double L, int N) {
    if (N <= 0) return NULL;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)N, sizeof(TriadCplx));
    if (!psi) return NULL;

    double target_ipr = (value - calib.b) / calib.a;
    if (target_ipr < 1e-3) target_ipr = 1e-3;

    double w = 1.0 / (target_ipr * sqrt(2.0 * TRIAD_PI));

    double w_hi = L / 4.0;
    if (w < 0.05) w = 0.05;
    if (w > w_hi) w = w_hi;

    double dx = _grid_dx(L, N);
    double denom = 2.0 * w * w;
    for (int i = 0; i < N; ++i) {
        double x = _grid_x(i, L, N);
        psi[i].re = exp(-(x*x) / denom);
        psi[i].im = 0.0;
    }

    _normalise(psi, N, dx);
    return psi;
}

double triad_decode_float(const TriadCplx *psi, int N, double dx,
                          TriadFloatCalib calib) {
    double s2 = 0.0, s4 = 0.0;
    for (int i = 0; i < N; ++i) {
        double r2 = psi[i].re * psi[i].re + psi[i].im * psi[i].im;
        s2 += r2;
        s4 += r2 * r2;
    }
    double n2 = s2 * dx;
    double n4 = s4 * dx;
    double denom = n2 * n2;
    if (denom < 1e-30) denom = 1e-30;
    double ipr = n4 / denom;
    return calib.a * ipr + calib.b;
}

TriadCplx *triad_encode_bits(long long value, int bit_width,
                             TriadIntCalib calib, double L, int N) {
    (void)bit_width;

    return triad_encode_int(value, calib, L, N, -1.0, 0.9);
}
