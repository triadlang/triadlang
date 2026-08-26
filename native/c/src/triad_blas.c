#include "triad_rt.h"
#include <math.h>
#include <string.h>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

void triad_vec_add(int64_t n, const double *a, const double *b, double *out) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void triad_vec_sub(int64_t n, const double *a, const double *b, double *out) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] - b[i];
}

void triad_vec_mul(int64_t n, const double *a, const double *b, double *out) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] * b[i];
}

void triad_vec_scale(int64_t n, double s, const double *a, double *out) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] * s;
}

double triad_vec_dot(int64_t n, const double *a, const double *b) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

void triad_vec_relu(int64_t n, const double *a, double *out) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] > 0.0 ? a[i] : 0.0;
}

void triad_vec_sigmoid(int64_t n, const double *a, double *out) {
    for (int64_t i = 0; i < n; i++) out[i] = 1.0 / (1.0 + exp(-a[i]));
}

void triad_vec_tanh(int64_t n, const double *a, double *out) {
    for (int64_t i = 0; i < n; i++) out[i] = tanh(a[i]);
}

void triad_matmul(int64_t M, int64_t K, int64_t N,
                  const double *A, const double *B, double *C) {
#ifdef USE_CBLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0, A, (int)K, B, (int)N, 0.0, C, (int)N);
#else
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            double s = 0.0;
            for (int64_t p = 0; p < K; p++)
                s += A[i * K + p] * B[p * N + j];
            C[i * N + j] = s;
        }
    }
#endif
}

void triad_blas_triad(int64_t batch, int64_t in_dim, int64_t out_dim,
                       const double *input, const double *weight,
                       const double *bias, double *output) {
    triad_matmul(batch, in_dim, out_dim, input, weight, output);
    if (bias) {
        for (int64_t i = 0; i < batch; i++)
            for (int64_t j = 0; j < out_dim; j++)
                output[i * out_dim + j] += bias[j];
    }
}
