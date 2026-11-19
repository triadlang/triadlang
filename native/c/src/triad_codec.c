/*
 * TriadLang Native Runtime — Physical codecs (Part II).
 *
 * Byte-for-byte port of runtime/codec.py:
 *   encode_int / decode_int   ↔ k* of Psi
 *   encode_bool / decode_bool ↔ crystallinity of Psi
 *   encode_float / decode_float ↔ IPR of Psi
 *   encode_bits delegates to encode_int.
 *
 * P1+P2+P3 invariance: codecs only prepare/read Psi; the solver that
 * consumes these arrays still runs the full equation
 *   i ℏ ∂_t Ψ = [-ℏ²/(2m)D² + V_ext + Λ|Ψ|² + V_mem + α(-Δ)^(σ/2) - iΓ] Ψ + η
 * with V_mem = Σ_j λ_j y_j and FDT-locked η. Codecs never amputate that.
 */
#include "triad_codec.h"
#include "triad_observables.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Match numpy.linspace(-L/2, L/2, N, endpoint=False):
 *   step = L/N, x[i] = -L/2 + i*step. dx = step. */
static inline double _grid_dx(double L, int N) {
    return L / (double)N;
}

static inline double _grid_x(int i, double L, int N) {
    return -0.5 * L + (double)i * (L / (double)N);
}

/* sum(|psi|^2)*dx — matches np.sum(np.abs(psi)**2) * dx */
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

/* ═══════════════════════════════════════════════════════════════════
   encode_int — psi(x) = G(x) * (1 + d * cos(k_n * x))
   Matches runtime/codec.py:encode_int exactly.
   ═══════════════════════════════════════════════════════════════════ */
TriadCplx *triad_encode_int(long long value, TriadIntCalib calib,
                            double L, int N,
                            double envelope_width, double modulation) {
    if (N <= 0) return NULL;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)N, sizeof(TriadCplx));
    if (!psi) return NULL;

    double dx = _grid_dx(L, N);
    double k_n = calib.k_min + (double)value * calib.k_step;

    /* default envelope_width = L/4 (Python: if envelope_width is None) */
    if (envelope_width <= 0.0) envelope_width = L / 4.0;

    if (k_n == 0.0) {
        /* python codec.py: if value == 0 or k_n == 0, psi = np.ones(N).
         * uniform constant before normalization (no envelope). */
        for (int i = 0; i < N; ++i) {
            psi[i].re = 1.0;
            psi[i].im = 0.0;
        }
    } else {
        /* psi = (1 + modulation * cos(k_n * x)), no envelope.
         * matches python codec.py:encode_int line 26 exactly. */
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

/* ═══════════════════════════════════════════════════════════════════
   decode_int — k* of psi, rounded to integer steps of calib.k_step.
   Uses k_min_exclude = 2*pi/L to skip DC bin (matches codec.py).
   ═══════════════════════════════════════════════════════════════════ */
long long triad_decode_int(const TriadCplx *psi, int N, double dx,
                           TriadIntCalib calib) {
    /* matches python codec.py:decode_int exactly:
     *   P = |fft(psi)|^2
     *   half = P[1 : N/2]
     *   vbin = argmax(half) + 1
     *   if P[vbin] < 1e-6 * P[0]: return 0
     *   k_star = vbin * (2*pi/L)
     *   return round((k_star - k_min) / k_step)
     */
    if (N < 4) return 0;
    double L = (double)N * dx;

    /* compute fft(psi) and then |psi_hat|^2 in native fft bin order
     * (bin 0 = DC, bins 1..N/2-1 = positive freqs).
     */
    TriadCplx *psi_hat = (TriadCplx *)malloc(sizeof(TriadCplx) * N);
    triad_fft_fn(N, psi, psi_hat);
    double *P = (double *)malloc(sizeof(double) * N);
    for (int i = 0; i < N; ++i)
        P[i] = psi_hat[i].re * psi_hat[i].re + psi_hat[i].im * psi_hat[i].im;
    free(psi_hat);

    int half_end = N / 2;  /* exclusive, matches python P[1:N//2] */
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

/* ═══════════════════════════════════════════════════════════════════
   encode_bool
     True  -> tight Gaussian (exp(-x^2/2))
     False -> broad Gaussian (exp(-x^2 / (2*(L/3)^2)))
   ═══════════════════════════════════════════════════════════════════ */
TriadCplx *triad_encode_bool(int value, double L, int N) {
    if (N <= 0) return NULL;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)N, sizeof(TriadCplx));
    if (!psi) return NULL;

    double dx = _grid_dx(L, N);

    if (value) {
        /* python codec.py: env = exp(-x^2 / (2*(L/6)^2)); psi = env * cos(3*x) */
        double w = L / 6.0;
        double denom = 2.0 * w * w;
        for (int i = 0; i < N; ++i) {
            double x = _grid_x(i, L, N);
            double env = exp(-(x*x) / denom);
            psi[i].re = env * cos(3.0 * x);
            psi[i].im = 0.0;
        }
    } else {
        /* python: psi = exp(-x^2 / (2*(L/3)^2)) */
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

/* ═══════════════════════════════════════════════════════════════════
   decode_bool — tri-state via crystallinity threshold.
     C > 0.5  -> TRIAD_BOOL_TRUE
     C < 0.2  -> TRIAD_BOOL_FALSE
     else     -> TRIAD_BOOL_AMBIGUOUS
   crystallinity() uses k_cutoff = 1.0 (Python default).
   ═══════════════════════════════════════════════════════════════════ */
int triad_decode_bool(const TriadCplx *psi, int N, double dx) {
    double c = triad_obs_crystallinity(psi, N, dx, 1.0);
    if (c > 0.5) return TRIAD_BOOL_TRUE;
    if (c < 0.2) return TRIAD_BOOL_FALSE;
    return TRIAD_BOOL_AMBIGUOUS;
}

/* ═══════════════════════════════════════════════════════════════════
   encode_float — Gaussian width tuned to hit target IPR.
     target_ipr = max((value - b)/a, 1e-3)
     w = 1 / (2 * sqrt(pi) * target_ipr), clipped to [0.05, L/4]
   ═══════════════════════════════════════════════════════════════════ */
TriadCplx *triad_encode_float(double value, TriadFloatCalib calib,
                              double L, int N) {
    if (N <= 0) return NULL;
    TriadCplx *psi = (TriadCplx *)calloc((size_t)N, sizeof(TriadCplx));
    if (!psi) return NULL;

    double target_ipr = (value - calib.b) / calib.a;
    if (target_ipr < 1e-3) target_ipr = 1e-3;

    /* python codec.py: w = 1.0 / (target_ipr * sqrt(2 * pi)) */
    double w = 1.0 / (target_ipr * sqrt(2.0 * TRIAD_PI));
    /* np.clip(w, 0.05, L/4) */
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

/* ═══════════════════════════════════════════════════════════════════
   decode_float — IPR(psi) -> a * IPR + b.
     IPR = sum(|psi|^4)*dx / (sum(|psi|^2)*dx)^2
   Matches codec.py:decode_float (inline, not via obs.ipr, to bit-match
   the floor 1e-30 on the denominator).
   ═══════════════════════════════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════════════════════════════
   encode_bits — delegate to encode_int. bit_width is bookkeeping only.
   ═══════════════════════════════════════════════════════════════════ */
TriadCplx *triad_encode_bits(long long value, int bit_width,
                             TriadIntCalib calib, double L, int N) {
    (void)bit_width;
    /* default envelope_width = L/4, default modulation = 0.9
       (encode_int with envelope_width<=0 picks L/4) */
    return triad_encode_int(value, calib, L, N, -1.0, 0.9);
}
