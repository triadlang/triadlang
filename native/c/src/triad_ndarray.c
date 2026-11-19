/*
 * TriadLang Native Runtime — NDArray (n-dimensional array, row-major)
 */
#include "triad_rt.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

static int64_t compute_size(int32_t ndim, int32_t *shape) {
    int64_t s = 1;
    for (int32_t i = 0; i < ndim; i++) s *= shape[i];
    return s;
}

static int32_t *copy_shape(int32_t ndim, int32_t *shape) {
    int32_t *s = malloc(sizeof(int32_t) * ndim);
    memcpy(s, shape, sizeof(int32_t) * ndim);
    return s;
}

static int32_t *compute_strides(int32_t ndim, int32_t *shape) {
    int32_t *strides = malloc(sizeof(int32_t) * ndim);
    strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; i--)
        strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
}

TriadNDArray *triad_ndarray_new(int32_t ndim, int32_t *shape) {
    TriadNDArray *a = malloc(sizeof(TriadNDArray));
    a->refcount = 1;
    a->ndim = ndim;
    a->shape = copy_shape(ndim, shape);
    a->strides = compute_strides(ndim, shape);
    a->size = compute_size(ndim, shape);
    a->data = calloc(a->size > 0 ? (size_t)a->size : 1, sizeof(double));
    a->owns_data = true;
    return a;
}

TriadNDArray *triad_ndarray_new_data(int32_t ndim, int32_t *shape, double *data) {
    TriadNDArray *a = triad_ndarray_new(ndim, shape);
    if (data && a->size > 0) memcpy(a->data, data, sizeof(double) * a->size);
    return a;
}

TriadNDArray *triad_ndarray_zeros(int32_t ndim, int32_t *shape) {
    return triad_ndarray_new(ndim, shape);
}

TriadNDArray *triad_ndarray_ones(int32_t ndim, int32_t *shape) {
    TriadNDArray *a = triad_ndarray_new(ndim, shape);
    for (int64_t i = 0; i < a->size; i++) a->data[i] = 1.0;
    return a;
}

void triad_ndarray_free(TriadNDArray *a) {
    if (!a) return;
    if (a->owns_data && a->data) free(a->data);
    if (a->shape) free(a->shape);
    if (a->strides) free(a->strides);
    free(a);
}

TriadNDArray *triad_ndarray_reshape(TriadNDArray *a, int32_t ndim, int32_t *shape) {
    if (!a) return NULL;
    int64_t newsize = compute_size(ndim, shape);
    if (newsize != a->size) return NULL;
    TriadNDArray *r = malloc(sizeof(TriadNDArray));
    r->refcount = 1;
    r->ndim = ndim;
    r->shape = copy_shape(ndim, shape);
    r->strides = compute_strides(ndim, shape);
    r->size = a->size;
    r->data = a->data;
    r->owns_data = false;
    return r;
}

TriadNDArray *triad_ndarray_slice(TriadNDArray *a, int32_t *starts, int32_t *ends, int32_t *steps) {
    if (!a) return NULL;
    int32_t out_shape[8];
    int64_t out_size = 1;
    for (int32_t d = 0; d < a->ndim; d++) {
        int32_t s = starts[d], e = ends[d], st = steps[d];
        if (s < 0) s += a->shape[d];
        if (e < 0) e += a->shape[d];
        if (s < 0) s = 0;
        if (s > a->shape[d]) s = a->shape[d];
        if (e < 0) e = 0;
        if (e > a->shape[d]) e = a->shape[d];
        int32_t len = (st > 0) ? (e - s + st - 1) / st : (s - e + (-st) - 1) / (-st);
        if (len < 0) len = 0;
        out_shape[d] = len;
        out_size *= len;
    }
    TriadNDArray *r = triad_ndarray_new(a->ndim, out_shape);
    int64_t idx[8] = {0};
    for (int64_t i = 0; i < out_size; i++) {
        int64_t src_flat = 0;
        int64_t tmp = i;
        for (int32_t d = a->ndim - 1; d >= 0; d--) {
            idx[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        for (int32_t d = 0; d < a->ndim; d++) {
            int64_t src_idx = starts[d] + idx[d] * steps[d];
            src_flat += src_idx * a->strides[d];
        }
        r->data[i] = a->data[src_flat];
    }
    return r;
}

int64_t triad_ndarray_size(TriadNDArray *a) {
    return a ? a->size : 0;
}

double triad_ndarray_get(TriadNDArray *a, int32_t *indices) {
    if (!a) return 0.0;
    int64_t flat = 0;
    for (int32_t d = 0; d < a->ndim; d++)
        flat += (int64_t)indices[d] * a->strides[d];
    return a->data[flat];
}

void triad_ndarray_set(TriadNDArray *a, int32_t *indices, double val) {
    if (!a) return;
    int64_t flat = 0;
    for (int32_t d = 0; d < a->ndim; d++)
        flat += (int64_t)indices[d] * a->strides[d];
    a->data[flat] = val;
}

double triad_ndarray_sum(TriadNDArray *a) {
    if (!a) return 0.0;
    double s = 0.0;
    for (int64_t i = 0; i < a->size; i++) s += a->data[i];
    return s;
}

double triad_ndarray_mean(TriadNDArray *a) {
    if (!a || a->size == 0) return 0.0;
    return triad_ndarray_sum(a) / (double)a->size;
}

double triad_ndarray_max(TriadNDArray *a) {
    if (!a || a->size == 0) return 0.0;
    double m = -DBL_MAX;
    for (int64_t i = 0; i < a->size; i++) if (a->data[i] > m) m = a->data[i];
    return m;
}

double triad_ndarray_min(TriadNDArray *a) {
    if (!a || a->size == 0) return 0.0;
    double m = DBL_MAX;
    for (int64_t i = 0; i < a->size; i++) if (a->data[i] < m) m = a->data[i];
    return m;
}
