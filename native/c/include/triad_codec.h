#ifndef TRIAD_CODEC_H
#define TRIAD_CODEC_H

#include "triad_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double k_min;
    double k_step;
} TriadIntCalib;

typedef struct {
    double a;
    double b;
} TriadFloatCalib;

#define TRIAD_DEFAULT_K_MIN   0.0
#define TRIAD_DEFAULT_K_STEP  0.196349541

#define TRIAD_BOOL_FALSE      0
#define TRIAD_BOOL_TRUE       1
#define TRIAD_BOOL_AMBIGUOUS (-1)

TriadCplx *triad_encode_int(long long value, TriadIntCalib calib,
                            double L, int N,
                            double envelope_width, double modulation);

TriadCplx *triad_encode_bool(int value, double L, int N);

TriadCplx *triad_encode_float(double value, TriadFloatCalib calib,
                              double L, int N);

TriadCplx *triad_encode_bits(long long value, int bit_width,
                             TriadIntCalib calib, double L, int N);

long long triad_decode_int(const TriadCplx *psi, int N, double dx,
                           TriadIntCalib calib);

int       triad_decode_bool(const TriadCplx *psi, int N, double dx);

double    triad_decode_float(const TriadCplx *psi, int N, double dx,
                             TriadFloatCalib calib);

#ifdef __cplusplus
}
#endif

#endif
