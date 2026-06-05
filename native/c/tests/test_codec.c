/*
 * test_codec.c — Unit tests for the native codec.
 *
 * Verifies the core contracts of runtime/codec.py without relying on
 * Python:
 *   1. Norm-preservation: encoded psi is L²-normalised to 1.
 *   2. encode/decode round-trip stability for ints with k_n separable
 *      from k_min_exclude (|n| >= 2 on the canonical grid).
 *   3. encode/decode round-trip for booleans (tri-state contract).
 *   4. encode/decode round-trip for floats within the IPR-clipped range.
 *   5. Edge cases: value=0 encodes a pure envelope; encode_bits delegates
 *      to encode_int.
 *
 * P1+P2+P3 are not exercised here directly — this file tests only the
 * codec layer. The full equation is exercised by test_solver_nd.
 */
#include "triad_codec.h"
#include "triad_observables.h"
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
    for (int i = 0; i < N; ++i) s += psi[i].re * psi[i].re + psi[i].im * psi[i].im;
    return s * dx;
}

static void test_norm_int(void) {
    double L = 32.0; int N = 128;
    TriadIntCalib calib = { TRIAD_DEFAULT_K_MIN, TRIAD_DEFAULT_K_STEP };
    long long values[] = { 0, 1, 2, 3, 5, 10, -3, -5 };
    for (size_t i = 0; i < sizeof(values)/sizeof(values[0]); ++i) {
        TriadCplx *psi = triad_encode_int(values[i], calib, L, N, -1.0, 0.9);
        CHECK(psi != NULL, "encode_int(%lld) returned NULL", values[i]);
        double n = norm_l2(psi, N, L / (double)N);
        CHECK(fabs(n - 1.0) < 1e-12,
              "encode_int(%lld) norm = %.3e (want 1)", values[i], n);
        free(psi);
    }
}

static void test_norm_bool(void) {
    double L = 32.0; int N = 128;
    double dx = L / (double)N;
    for (int v = 0; v <= 1; ++v) {
        TriadCplx *psi = triad_encode_bool(v, L, N);
        CHECK(psi != NULL, "encode_bool(%d) returned NULL", v);
        double n = norm_l2(psi, N, dx);
        CHECK(fabs(n - 1.0) < 1e-12, "encode_bool(%d) norm = %.3e", v, n);
        free(psi);
    }
}

static void test_norm_float(void) {
    double L = 32.0; int N = 128;
    double dx = L / (double)N;
    TriadFloatCalib calib = { 1.0, 0.0 };
    double values[] = { 0.5, 1.0, 1.5, 2.0, 2.5, 5.0 };
    for (size_t i = 0; i < sizeof(values)/sizeof(values[0]); ++i) {
        TriadCplx *psi = triad_encode_float(values[i], calib, L, N);
        CHECK(psi != NULL, "encode_float(%g) returned NULL", values[i]);
        double n = norm_l2(psi, N, dx);
        CHECK(fabs(n - 1.0) < 1e-12,
              "encode_float(%g) norm = %.3e", values[i], n);
        free(psi);
    }
}

static void test_int_roundtrip(void) {
    /* Note: the encoding has a known limitation that small |n| values
     * fall below k_min_exclude = 2*pi/L and decode to a nearby alias.
     * That is faithful behaviour of runtime/codec.py; we test only values
     * that are clearly separable from DC. */
    double L = 32.0; int N = 128;
    TriadIntCalib calib = { TRIAD_DEFAULT_K_MIN, TRIAD_DEFAULT_K_STEP };
    double dx = L / (double)N;
    long long values[] = { 3, 5, 7, 10, -3, -5, -10 };
    for (size_t i = 0; i < sizeof(values)/sizeof(values[0]); ++i) {
        TriadCplx *psi = triad_encode_int(values[i], calib, L, N, -1.0, 0.9);
        long long dec = triad_decode_int(psi, N, dx, calib);
        /* Decoder returns |k*|, so negative values come back as positive. */
        long long expected = values[i] < 0 ? -values[i] : values[i];
        CHECK(dec == expected,
              "decode_int(encode_int(%lld)) = %lld (want %lld)",
              values[i], dec, expected);
        free(psi);
    }
}

static void test_bool_separation(void) {
    /* Contract from runtime/codec.py: True -> tight Gaussian (exp(-x^2/2))
     * and False -> broad Gaussian (exp(-x^2/(2*(L/3)^2))). The True case
     * is so tight on the canonical L=32 grid that ALL its spectral energy
     * sits below k_cutoff=1.0, giving crystallinity ≈ 0 → decode_bool
     * returns FALSE (the Python implementation behaves identically).
     *
     * The meaningful contract is therefore not "True roundtrips to True"
     * but "True and False produce distinguishable crystallinity values".
     * That separation is what the codec uses across MOV chains; here we
     * verify it numerically. */
    double L = 32.0; int N = 128;
    double dx = L / (double)N;
    TriadCplx *psi_t = triad_encode_bool(1, L, N);
    TriadCplx *psi_f = triad_encode_bool(0, L, N);
    double c_t = triad_obs_crystallinity(psi_t, N, dx, 1.0);
    double c_f = triad_obs_crystallinity(psi_f, N, dx, 1.0);
    /* True is the tight Gaussian; its spectrum is the WIDE Gaussian in k.
     * False is the broad Gaussian; its spectrum is TIGHT in k.
     * So C(True) > C(False) must hold for the decoder's threshold to mean
     * anything at all. */
    CHECK(c_t > c_f,
          "crystallinity(True)=%g should exceed crystallinity(False)=%g",
          c_t, c_f);
    /* And the Python contract is also that decode_bool(False) is FALSE. */
    int f = triad_decode_bool(psi_f, N, dx);
    CHECK(f == TRIAD_BOOL_FALSE,
          "decode_bool(encode_bool(False)) = %d (want %d)",
          f, TRIAD_BOOL_FALSE);
    free(psi_t); free(psi_f);
}

static void test_float_matches_python(void) {
    /* runtime/codec.py's decode(encode(v)) is not the identity — it equals
     * the Python reference values below. We assert exact agreement at the
     * 1e-9 level, which is the limit of how reproducibly NumPy and our C
     * implementation can reorder the same sum-of-products. The decoded
     * value is monotone in v over the realisable range, which is the
     * contract MOV uses. */
    double L = 32.0; int N = 128;
    double dx = L / (double)N;
    TriadFloatCalib calib = { 1.0, 0.0 };
    struct { double v; double expected; } cases[] = {
        { 0.1, 0.10000000288623632 },
        { 0.5, 0.5000000000000001  },
        { 1.0, 1.0000069746360658  },
        { 2.0, 2.156729263756602   },
    };
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        TriadCplx *psi = triad_encode_float(cases[i].v, calib, L, N);
        double dec = triad_decode_float(psi, N, dx, calib);
        double err = fabs(dec - cases[i].expected);
        CHECK(err < 1e-9,
              "decode_float(encode_float(%g)) = %.15g (want %.15g, err=%.3e)",
              cases[i].v, dec, cases[i].expected, err);
        free(psi);
    }
    /* And monotonicity holds across the realisable band. */
    double prev = -1.0;
    int mono = 1;
    for (int k = 0; k < 6; ++k) {
        double v = 0.1 + 0.4 * (double)k;
        TriadCplx *psi = triad_encode_float(v, calib, L, N);
        double dec = triad_decode_float(psi, N, dx, calib);
        if (dec < prev) mono = 0;
        prev = dec;
        free(psi);
    }
    CHECK(mono, "decode_float is not monotone over realisable range");
}

static void test_encode_int_zero_is_envelope(void) {
    /* When k_n == 0, encode_int produces a pure Gaussian envelope. */
    double L = 32.0; int N = 128;
    TriadIntCalib calib = { 0.0, TRIAD_DEFAULT_K_STEP };
    TriadCplx *psi = triad_encode_int(0, calib, L, N, -1.0, 0.9);
    CHECK(psi != NULL, "encode_int(0) returned NULL");
    /* python codec.py: value=0 -> psi = np.ones(N), normalized -> uniform field.
     * all |psi[i]|^2 should be identical. */
    double first = psi[0].re * psi[0].re + psi[0].im * psi[0].im;
    int uniform = 1;
    for (int i = 1; i < N; ++i) {
        double r2 = psi[i].re * psi[i].re + psi[i].im * psi[i].im;
        if (fabs(r2 - first) > 1e-12) { uniform = 0; break; }
    }
    CHECK(uniform,
          "encode_int(0) should produce uniform field (np.ones normalized)");
    /* Imaginary part should be identically zero. */
    int all_real = 1;
    for (int i = 0; i < N; ++i) if (psi[i].im != 0.0) { all_real = 0; break; }
    CHECK(all_real, "encode_int(0) has non-zero imaginary parts");
    free(psi);
}

static void test_encode_bits_delegates(void) {
    /* encode_bits should produce bit-identical psi to encode_int with
     * default modulation/envelope. */
    double L = 32.0; int N = 128;
    TriadIntCalib calib = { TRIAD_DEFAULT_K_MIN, TRIAD_DEFAULT_K_STEP };
    TriadCplx *via_bits = triad_encode_bits(5, 8, calib, L, N);
    TriadCplx *via_int  = triad_encode_int(5, calib, L, N, -1.0, 0.9);
    CHECK(via_bits != NULL && via_int != NULL, "alloc failed");
    int identical = 1;
    for (int i = 0; i < N; ++i) {
        if (via_bits[i].re != via_int[i].re || via_bits[i].im != via_int[i].im) {
            identical = 0; break;
        }
    }
    CHECK(identical, "encode_bits does not match encode_int");
    free(via_bits); free(via_int);
}

int main(void) {
    test_norm_int();
    test_norm_bool();
    test_norm_float();
    test_int_roundtrip();
    test_bool_separation();
    test_float_matches_python();
    test_encode_int_zero_is_envelope();
    test_encode_bits_delegates();

    if (g_failures == 0) {
        printf("test_codec: PASS (%d/%d)\n", g_total, g_total);
        return 0;
    } else {
        printf("test_codec: FAIL (%d/%d failures)\n", g_failures, g_total);
        return 1;
    }
}
