#ifndef TRIAD_CODEC_H
#define TRIAD_CODEC_H

/* ════════════════════════════════════════════════════════════════════
   TriadLang — Physical codecs (Part II)
   Native C port of runtime/codec.py.

   Maps program values to/from substrate state:
     int   <-> dominant wavenumber k*       (§2.1)
     bool  <-> crystallinity C(T)           (§2.2)
     float <-> IPR(T)                       (§2.3)

   Conventions (mirror runtime/codec.py exactly):
     int  : k* = k_min + n * k_step
     bool : True  if  C > 0.5
            False if  C < 0.2
            ambiguous in between
     float: x = a * IPR + b
   ════════════════════════════════════════════════════════════════════ */

#include "triad_rt.h"   /* TriadCplx */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Calibrations ───────────────────────────────────────────────── */

typedef struct {
    double k_min;
    double k_step;
} TriadIntCalib;

typedef struct {
    double a;
    double b;
} TriadFloatCalib;

/* Default scale: matches DEFAULT_K_MIN / DEFAULT_K_STEP in codec.py.
   k_step = 2*pi/32 = smallest non-zero k bin on canonical L=32 grid. */
#define TRIAD_DEFAULT_K_MIN   0.0
#define TRIAD_DEFAULT_K_STEP  0.196349541

/* tri-state for decode_bool */
#define TRIAD_BOOL_FALSE      0
#define TRIAD_BOOL_TRUE       1
#define TRIAD_BOOL_AMBIGUOUS (-1)

/* ── Encoders (caller frees with free()) ────────────────────────── */

/* envelope_width <= 0  => default L/4
   modulation         => default 0.9 (pass 0.9 explicitly to match Python) */
TriadCplx *triad_encode_int(long long value, TriadIntCalib calib,
                            double L, int N,
                            double envelope_width, double modulation);

TriadCplx *triad_encode_bool(int value, double L, int N);

TriadCplx *triad_encode_float(double value, TriadFloatCalib calib,
                              double L, int N);

/* encode_bits delegates to encode_int — bit_width is bookkeeping. */
TriadCplx *triad_encode_bits(long long value, int bit_width,
                             TriadIntCalib calib, double L, int N);

/* ── Decoders ───────────────────────────────────────────────────── */

long long triad_decode_int(const TriadCplx *psi, int N, double dx,
                           TriadIntCalib calib);

/* returns TRIAD_BOOL_TRUE / TRIAD_BOOL_FALSE / TRIAD_BOOL_AMBIGUOUS */
int       triad_decode_bool(const TriadCplx *psi, int N, double dx);

double    triad_decode_float(const TriadCplx *psi, int N, double dx,
                             TriadFloatCalib calib);

#ifdef __cplusplus
}
#endif

#endif /* TRIAD_CODEC_H */
