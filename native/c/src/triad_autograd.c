#include "triad_ml.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int _triad_grad_enabled = 1;

void triad_ml_set_grad(int on) { _triad_grad_enabled = on; }
int  triad_ml_get_grad(void)   { return _triad_grad_enabled; }

static uint64_t _ml_rng_state = 123456789ULL;

void triad_ml_seed(uint64_t s) { _ml_rng_state = s ? s : 1ULL; }

static uint64_t _xorshift64(void) {
    uint64_t x = _ml_rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    _ml_rng_state = x;
    return x;
}

static double _randf(void) {
    return (_xorshift64() >> 11) * (1.0 / 9007199254740992.0);
}

static double _randn(void) {
    double u1 = _randf(), u2 = _randf();
    if (u1 < 1e-30) u1 = 1e-30;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void _ctx_free_plain(void *ctx) {
    free(ctx);
}

TriadTensor *triad_tensor_new(int32_t ndim, const int32_t *shape, int requires_grad) {
    if (ndim < 1 || !shape) return NULL;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 1) return NULL;
    }
    TriadTensor *t = calloc(1, sizeof(TriadTensor));
    if (!t) return NULL;
    t->ndim = ndim;
    t->shape = malloc((size_t)ndim * sizeof(int32_t));
    if (!t->shape) { free(t); return NULL; }
    memcpy(t->shape, shape, (size_t)ndim * sizeof(int32_t));
    int64_t sz = 1;
    for (int i = 0; i < ndim; i++) sz *= shape[i];
    t->size = sz;
    t->data = calloc((size_t)sz, sizeof(double));
    if (!t->data) { free(t->shape); free(t); return NULL; }
    t->grad = NULL;
    t->requires_grad = requires_grad;
    t->grad_fn = NULL;
    t->nchildren = 0;
    t->children = NULL;
    t->refcount = 1;
    return t;
}

TriadTensor *triad_tensor_from_data(int32_t ndim, const int32_t *shape,
                                     const double *data, int requires_grad) {
    TriadTensor *t = triad_tensor_new(ndim, shape, requires_grad);
    memcpy(t->data, data, t->size * sizeof(double));
    return t;
}

TriadTensor *triad_tensor_scalar(double val, int requires_grad) {
    int32_t shape[] = {1};
    TriadTensor *t = triad_tensor_new(1, shape, requires_grad);
    if (!t) return NULL;
    t->size = 1;
    t->data[0] = val;

    free(t->shape);
    t->shape = NULL;
    t->ndim = 0;
    return t;
}

TriadTensor *triad_tensor_zeros(int32_t ndim, const int32_t *shape, int rg) {
    return triad_tensor_new(ndim, shape, rg);
}

TriadTensor *triad_tensor_ones(int32_t ndim, const int32_t *shape, int rg) {
    TriadTensor *t = triad_tensor_new(ndim, shape, rg);
    for (int64_t i = 0; i < t->size; i++) t->data[i] = 1.0;
    return t;
}

TriadTensor *triad_tensor_randn(int32_t ndim, const int32_t *shape, int rg) {
    TriadTensor *t = triad_tensor_new(ndim, shape, rg);
    for (int64_t i = 0; i < t->size; i++) t->data[i] = _randn();
    return t;
}

TriadTensor *triad_tensor_rand(int32_t ndim, const int32_t *shape, int rg) {
    TriadTensor *t = triad_tensor_new(ndim, shape, rg);
    for (int64_t i = 0; i < t->size; i++) t->data[i] = _randf();
    return t;
}

TriadTensor *triad_tensor_eye(int32_t n) {
    int32_t shape[] = {n, n};
    TriadTensor *t = triad_tensor_new(2, shape, 0);
    for (int32_t i = 0; i < n; i++) t->data[i * n + i] = 1.0;
    return t;
}

TriadTensor *triad_tensor_arange(double start, double stop, double step) {
    int32_t n = (int32_t)ceil((stop - start) / step);
    if (n < 0) n = 0;
    int32_t shape[] = {n};
    TriadTensor *t = triad_tensor_new(1, shape, 0);
    for (int32_t i = 0; i < n; i++) t->data[i] = start + i * step;
    return t;
}

TriadTensor *triad_tensor_linspace(double start, double stop, int32_t steps) {
    if (steps < 0) steps = 0;
    int32_t shape[] = {steps};
    TriadTensor *t = triad_tensor_new(1, shape, 0);
    if (steps == 1) {
        t->data[0] = start;
    } else if (steps > 1) {
        double step = (stop - start) / (double)(steps - 1);
        for (int32_t i = 0; i < steps; i++) t->data[i] = start + step * (double)i;
    }
    return t;
}

void triad_tensor_free(TriadTensor *t) {
    if (!t) return;
    t->refcount--;
    if (t->refcount > 0) return;
    if (t->ctx_free && t->_ctx) t->ctx_free(t->_ctx);
    for (int32_t i = 0; i < t->nchildren; i++) triad_tensor_free(t->children[i]);
    free(t->data);
    free(t->grad);
    free(t->shape);
    free(t->children);
    free(t);
}

void triad_tensor_retain(TriadTensor *t) { if (t) t->refcount++; }

static void _ensure_grad(TriadTensor *t) {
    if (!t->grad) {
        t->grad = calloc(t->size, sizeof(double));
    }
}

static void _accumulate_grad(TriadTensor *t, const double *g) {
    _ensure_grad(t);
    for (int64_t i = 0; i < t->size; i++) t->grad[i] += g[i];
}

static int64_t _shape_size(int32_t ndim, const int32_t *shape) {
    int64_t size = 1;
    for (int32_t i = 0; i < ndim; i++) size *= shape[i];
    return size;
}

static int32_t _norm_axis(int32_t axis, int32_t ndim) {
    if (axis < 0) axis += ndim;
    return axis;
}

static double *_unbroadcast(const double *g, int32_t g_ndim, const int32_t *g_shape,
                            int32_t t_ndim, const int32_t *t_shape, int64_t t_size) {

    if (g_ndim == t_ndim) {
        int same = 1;
        for (int i = 0; i < g_ndim && same; i++)
            if (g_shape[i] != t_shape[i]) same = 0;
        if (same) {
            double *out = malloc(t_size * sizeof(double));
            memcpy(out, g, t_size * sizeof(double));
            return out;
        }
    }

    double *out = calloc(t_size, sizeof(double));

    int64_t g_size = 1;
    for (int i = 0; i < g_ndim; i++) g_size *= g_shape[i];

    if (t_ndim == 0 || t_size == 1) {
        double s = 0.0;
        for (int64_t i = 0; i < g_size; i++) s += g[i];
        out[0] = s;
        return out;
    }

    int32_t padded[32];
    int pad = g_ndim - t_ndim;
    for (int i = 0; i < pad; i++) padded[i] = 1;
    for (int i = 0; i < t_ndim; i++) padded[pad + i] = t_shape[i];

    for (int64_t gi = 0; gi < g_size; gi++) {
        int64_t idx = gi;
        int64_t ti = 0;
        int64_t t_stride = 1;
        for (int d = g_ndim - 1; d >= 0; d--) {
            int32_t coord = idx % g_shape[d];
            idx /= g_shape[d];
            int32_t tc = (padded[d] == 1) ? 0 : coord;

            int64_t s = 1;
            for (int k = d + 1; k < g_ndim; k++) s *= padded[k];
            ti += tc * s;
            (void)t_stride;
        }

        int64_t coords[32];
        int64_t tmp = gi;
        for (int d = g_ndim - 1; d >= 0; d--) {
            coords[d] = tmp % g_shape[d];
            tmp /= g_shape[d];
        }
        ti = 0;
        int64_t s = 1;
        for (int d = t_ndim - 1; d >= 0; d--) {
            int32_t c = (padded[pad + d] == 1) ? 0 : (int32_t)coords[pad + d];
            ti += c * s;
            s *= t_shape[d];
        }
        out[ti] += g[gi];
    }
    return out;
}

static void _set_children(TriadTensor *out, int n, ...) {
    out->nchildren = n;
    out->children = malloc(n * sizeof(TriadTensor*));
    va_list ap;
    va_start(ap, n);
    for (int i = 0; i < n; i++) {
        out->children[i] = va_arg(ap, TriadTensor*);
        triad_tensor_retain(out->children[i]);
    }
    va_end(ap);
}

typedef struct {
    TriadTensor **topo;
    int64_t     *visited;
    int32_t      len;
    int32_t      nvisited;
    int32_t      cap;
} TopoBuild;

static int _topo_grow(TopoBuild *tb) {
    int32_t next = tb->cap ? tb->cap * 2 : 256;
    TriadTensor **topo = realloc(tb->topo, (size_t)next * sizeof(TriadTensor*));
    if (!topo) return 0;
    int64_t *visited = realloc(tb->visited, (size_t)next * sizeof(int64_t));
    if (!visited) return 0;
    tb->topo = topo;
    tb->visited = visited;
    tb->cap = next;
    return 1;
}

static int _topo_build(TriadTensor *t, TopoBuild *tb) {

    int64_t tid = (int64_t)(uintptr_t)t;
    for (int32_t i = 0; i < tb->nvisited; i++)
        if (tb->visited[i] == tid) return 1;
    if (tb->nvisited >= tb->cap && !_topo_grow(tb)) return 0;
    tb->visited[tb->nvisited++] = tid;
    for (int32_t i = 0; i < t->nchildren; i++) {
        if (!_topo_build(t->children[i], tb)) return 0;
    }
    if (tb->len >= tb->cap && !_topo_grow(tb)) return 0;
    tb->topo[tb->len++] = t;
    return 1;
}

void triad_tensor_backward(TriadTensor *t, const double *grad_out) {
    _ensure_grad(t);
    if (grad_out) {
        for (int64_t i = 0; i < t->size; i++) t->grad[i] += grad_out[i];
    } else {
        for (int64_t i = 0; i < t->size; i++) t->grad[i] = 1.0;
    }

    TopoBuild tb = {0};
    if (!_topo_grow(&tb) || !_topo_build(t, &tb)) {
        free(tb.topo);
        free(tb.visited);
        return;
    }

    for (int32_t i = tb.len - 1; i >= 0; i--) {
        if (tb.topo[i]->grad_fn && tb.topo[i]->grad) {
            tb.topo[i]->grad_fn(tb.topo[i]);
        }
    }
    free(tb.topo);
    free(tb.visited);
}

void triad_tensor_zero_grad(TriadTensor *t) {
    if (t->grad) {
        memset(t->grad, 0, t->size * sizeof(double));
    }
}

typedef struct {
    TriadTensor *a, *b;
    TriadTensor *out;
} BinaryCtx;

static void _add_backward(TriadTensor *out) {
    BinaryCtx *c = (BinaryCtx*)out->_ctx;
    if (c->a->requires_grad) {
        double *ug = _unbroadcast(out->grad, out->ndim, out->shape,
                                   c->a->ndim, c->a->shape, c->a->size);
        _accumulate_grad(c->a, ug);
        free(ug);
    }
    if (c->b->requires_grad) {
        double *ug = _unbroadcast(out->grad, out->ndim, out->shape,
                                   c->b->ndim, c->b->shape, c->b->size);
        _accumulate_grad(c->b, ug);
        free(ug);
    }
}

TriadTensor *triad_tensor_add(TriadTensor *a, TriadTensor *b) {
    if (!a || !b) return NULL;
    int32_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    int32_t shape[32];

    int32_t a_pad = ndim - a->ndim, b_pad = ndim - b->ndim;
    for (int i = 0; i < ndim; i++) {
        int32_t as = (i >= a_pad && a->shape) ? a->shape[i - a_pad] : 1;
        int32_t bs = (i >= b_pad && b->shape) ? b->shape[i - b_pad] : 1;
        if (as != bs && as != 1 && bs != 1) return NULL;
        shape[i] = as > bs ? as : bs;
    }
    TriadTensor *out = triad_tensor_new(ndim, shape, 0);

    for (int64_t oi = 0; oi < out->size; oi++) {

        int64_t tmp = oi;
        int32_t coords[32];
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d] = tmp % shape[d]; tmp /= shape[d];
        }
        int64_t ai = 0, bi = 0, as_stride = 1, bs_stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            int32_t adim = (d >= a_pad && a->shape) ? a->shape[d - a_pad] : 1;
            int32_t bdim = (d >= b_pad && b->shape) ? b->shape[d - b_pad] : 1;
            ai += (adim == 1 ? 0 : coords[d]) * as_stride;
            bi += (bdim == 1 ? 0 : coords[d]) * bs_stride;
            as_stride *= adim;
            bs_stride *= bdim;
        }
        out->data[oi] = a->data[ai] + b->data[bi];
    }

    if (_triad_grad_enabled && (a->requires_grad || b->requires_grad)) {
        out->requires_grad = 1;
        _set_children(out, 2, a, b);
        BinaryCtx *ctx = malloc(sizeof(BinaryCtx));
        ctx->a = a; ctx->b = b; ctx->out = out;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _add_backward;
    }
    return out;
}

static void _sub_backward(TriadTensor *out) {
    BinaryCtx *c = (BinaryCtx*)out->_ctx;
    if (c->a->requires_grad) {
        double *ug = _unbroadcast(out->grad, out->ndim, out->shape,
                                   c->a->ndim, c->a->shape, c->a->size);
        _accumulate_grad(c->a, ug);
        free(ug);
    }
    if (c->b->requires_grad) {
        double *ug = _unbroadcast(out->grad, out->ndim, out->shape,
                                   c->b->ndim, c->b->shape, c->b->size);

        for (int64_t i = 0; i < c->b->size; i++) ug[i] = -ug[i];
        _accumulate_grad(c->b, ug);
        free(ug);
    }
}

TriadTensor *triad_tensor_sub(TriadTensor *a, TriadTensor *b) {
    if (!a || !b) return NULL;
    int32_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    int32_t shape[32];
    int32_t a_pad = ndim - a->ndim, b_pad = ndim - b->ndim;
    for (int i = 0; i < ndim; i++) {
        int32_t as = (i >= a_pad && a->shape) ? a->shape[i - a_pad] : 1;
        int32_t bs = (i >= b_pad && b->shape) ? b->shape[i - b_pad] : 1;
        if (as != bs && as != 1 && bs != 1) return NULL;
        shape[i] = as > bs ? as : bs;
    }
    TriadTensor *out = triad_tensor_new(ndim, shape, 0);
    for (int64_t oi = 0; oi < out->size; oi++) {
        int64_t tmp = oi;
        int32_t coords[32];
        for (int d = ndim - 1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
        int64_t ai = 0, bi = 0, as_s = 1, bs_s = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            int32_t ad = (d >= a_pad && a->shape) ? a->shape[d - a_pad] : 1;
            int32_t bd = (d >= b_pad && b->shape) ? b->shape[d - b_pad] : 1;
            ai += (ad == 1 ? 0 : coords[d]) * as_s;
            bi += (bd == 1 ? 0 : coords[d]) * bs_s;
            as_s *= ad; bs_s *= bd;
        }
        out->data[oi] = a->data[ai] - b->data[bi];
    }
    if (_triad_grad_enabled && (a->requires_grad || b->requires_grad)) {
        out->requires_grad = 1;
        _set_children(out, 2, a, b);
        BinaryCtx *ctx = malloc(sizeof(BinaryCtx));
        ctx->a = a; ctx->b = b; ctx->out = out;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _sub_backward;
    }
    return out;
}

static void _mul_backward(TriadTensor *out) {
    BinaryCtx *c = (BinaryCtx*)out->_ctx;
    if (c->a->requires_grad) {
        double *tmp = malloc(out->size * sizeof(double));

        int32_t ndim = out->ndim;
        int32_t b_pad = ndim - c->b->ndim;
        for (int64_t oi = 0; oi < out->size; oi++) {
            int64_t idx = oi;
            int32_t coords[32];
            for (int d = ndim - 1; d >= 0; d--) { coords[d] = idx % out->shape[d]; idx /= out->shape[d]; }
            int64_t bi = 0, bs_s = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                int32_t bd = (d >= b_pad && c->b->shape) ? c->b->shape[d - b_pad] : 1;
                bi += (bd == 1 ? 0 : coords[d]) * bs_s;
                bs_s *= bd;
            }
            tmp[oi] = out->grad[oi] * c->b->data[bi];
        }
        double *ug = _unbroadcast(tmp, out->ndim, out->shape,
                                   c->a->ndim, c->a->shape, c->a->size);
        _accumulate_grad(c->a, ug);
        free(tmp); free(ug);
    }
    if (c->b->requires_grad) {
        double *tmp = malloc(out->size * sizeof(double));
        int32_t ndim = out->ndim;
        int32_t a_pad = ndim - c->a->ndim;
        for (int64_t oi = 0; oi < out->size; oi++) {
            int64_t idx = oi;
            int32_t coords[32];
            for (int d = ndim - 1; d >= 0; d--) { coords[d] = idx % out->shape[d]; idx /= out->shape[d]; }
            int64_t ai = 0, as_s = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                int32_t ad = (d >= a_pad && c->a->shape) ? c->a->shape[d - a_pad] : 1;
                ai += (ad == 1 ? 0 : coords[d]) * as_s;
                as_s *= ad;
            }
            tmp[oi] = out->grad[oi] * c->a->data[ai];
        }
        double *ug = _unbroadcast(tmp, out->ndim, out->shape,
                                   c->b->ndim, c->b->shape, c->b->size);
        _accumulate_grad(c->b, ug);
        free(tmp); free(ug);
    }
}

TriadTensor *triad_tensor_mul(TriadTensor *a, TriadTensor *b) {
    if (!a || !b) return NULL;
    int32_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    int32_t shape[32];
    int32_t a_pad = ndim - a->ndim, b_pad = ndim - b->ndim;
    for (int i = 0; i < ndim; i++) {
        int32_t as = (i >= a_pad && a->shape) ? a->shape[i - a_pad] : 1;
        int32_t bs = (i >= b_pad && b->shape) ? b->shape[i - b_pad] : 1;
        if (as != bs && as != 1 && bs != 1) return NULL;
        shape[i] = as > bs ? as : bs;
    }
    TriadTensor *out = triad_tensor_new(ndim, shape, 0);
    for (int64_t oi = 0; oi < out->size; oi++) {
        int64_t tmp = oi;
        int32_t coords[32];
        for (int d = ndim - 1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
        int64_t ai = 0, bi = 0, as_s = 1, bs_s = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            int32_t ad = (d >= a_pad && a->shape) ? a->shape[d - a_pad] : 1;
            int32_t bd = (d >= b_pad && b->shape) ? b->shape[d - b_pad] : 1;
            ai += (ad == 1 ? 0 : coords[d]) * as_s;
            bi += (bd == 1 ? 0 : coords[d]) * bs_s;
            as_s *= ad; bs_s *= bd;
        }
        out->data[oi] = a->data[ai] * b->data[bi];
    }
    if (_triad_grad_enabled && (a->requires_grad || b->requires_grad)) {
        out->requires_grad = 1;
        _set_children(out, 2, a, b);
        BinaryCtx *ctx = malloc(sizeof(BinaryCtx));
        ctx->a = a; ctx->b = b; ctx->out = out;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _mul_backward;
    }
    return out;
}

static void _div_backward(TriadTensor *out) {
    BinaryCtx *c = (BinaryCtx*)out->_ctx;
    int32_t ndim = out->ndim;
    int32_t a_pad = ndim - c->a->ndim, b_pad = ndim - c->b->ndim;
    if (c->a->requires_grad) {
        double *tmp = malloc(out->size * sizeof(double));
        for (int64_t oi = 0; oi < out->size; oi++) {
            int64_t idx = oi;
            int32_t coords[32];
            for (int d = ndim - 1; d >= 0; d--) { coords[d] = idx % out->shape[d]; idx /= out->shape[d]; }
            int64_t bi = 0, bs_s = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                int32_t bd = (d >= b_pad && c->b->shape) ? c->b->shape[d - b_pad] : 1;
                bi += (bd == 1 ? 0 : coords[d]) * bs_s;
                bs_s *= bd;
            }
            tmp[oi] = out->grad[oi] / c->b->data[bi];
        }
        double *ug = _unbroadcast(tmp, ndim, out->shape, c->a->ndim, c->a->shape, c->a->size);
        _accumulate_grad(c->a, ug);
        free(tmp); free(ug);
    }
    if (c->b->requires_grad) {
        double *tmp = malloc(out->size * sizeof(double));
        for (int64_t oi = 0; oi < out->size; oi++) {
            int64_t idx = oi;
            int32_t coords[32];
            for (int d = ndim - 1; d >= 0; d--) { coords[d] = idx % out->shape[d]; idx /= out->shape[d]; }
            int64_t ai = 0, bi = 0, as_s = 1, bs_s = 1;
            for (int d = ndim - 1; d >= 0; d--) {
                int32_t ad = (d >= a_pad && c->a->shape) ? c->a->shape[d - a_pad] : 1;
                int32_t bd = (d >= b_pad && c->b->shape) ? c->b->shape[d - b_pad] : 1;
                ai += (ad == 1 ? 0 : coords[d]) * as_s;
                bi += (bd == 1 ? 0 : coords[d]) * bs_s;
                as_s *= ad; bs_s *= bd;
            }
            tmp[oi] = -out->grad[oi] * c->a->data[ai] / (c->b->data[bi] * c->b->data[bi]);
        }
        double *ug = _unbroadcast(tmp, ndim, out->shape, c->b->ndim, c->b->shape, c->b->size);
        _accumulate_grad(c->b, ug);
        free(tmp); free(ug);
    }
}

TriadTensor *triad_tensor_div(TriadTensor *a, TriadTensor *b) {
    if (!a || !b) return NULL;
    int32_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    int32_t shape[32];
    int32_t a_pad = ndim - a->ndim, b_pad = ndim - b->ndim;
    for (int i = 0; i < ndim; i++) {
        int32_t as = (i >= a_pad && a->shape) ? a->shape[i - a_pad] : 1;
        int32_t bs = (i >= b_pad && b->shape) ? b->shape[i - b_pad] : 1;
        if (as != bs && as != 1 && bs != 1) return NULL;
        shape[i] = as > bs ? as : bs;
    }
    TriadTensor *out = triad_tensor_new(ndim, shape, 0);
    for (int64_t oi = 0; oi < out->size; oi++) {
        int64_t tmp = oi;
        int32_t coords[32];
        for (int d = ndim - 1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
        int64_t ai = 0, bi = 0, as_s = 1, bs_s = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            int32_t ad = (d >= a_pad && a->shape) ? a->shape[d - a_pad] : 1;
            int32_t bd = (d >= b_pad && b->shape) ? b->shape[d - b_pad] : 1;
            ai += (ad == 1 ? 0 : coords[d]) * as_s;
            bi += (bd == 1 ? 0 : coords[d]) * bs_s;
            as_s *= ad; bs_s *= bd;
        }
        out->data[oi] = a->data[ai] / b->data[bi];
    }
    if (_triad_grad_enabled && (a->requires_grad || b->requires_grad)) {
        out->requires_grad = 1;
        _set_children(out, 2, a, b);
        BinaryCtx *ctx = malloc(sizeof(BinaryCtx));
        ctx->a = a; ctx->b = b; ctx->out = out;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _div_backward;
    }
    return out;
}

typedef struct { TriadTensor *a; } UnaryCtx;

static void _neg_backward(TriadTensor *out) {
    UnaryCtx *c = (UnaryCtx*)out->_ctx;
    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        for (int64_t i = 0; i < c->a->size; i++)
            c->a->grad[i] -= out->grad[i];
    }
}

TriadTensor *triad_tensor_neg(TriadTensor *a) {
    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    for (int64_t i = 0; i < a->size; i++) out->data[i] = -a->data[i];
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        UnaryCtx *ctx = malloc(sizeof(UnaryCtx));
        ctx->a = a;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _neg_backward;
    }
    return out;
}

typedef struct { TriadTensor *a; double exp_val; } PowCtx;

static void _pow_backward(TriadTensor *out) {
    PowCtx *c = (PowCtx*)out->_ctx;
    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        for (int64_t i = 0; i < c->a->size; i++) {
            c->a->grad[i] += out->grad[i] * c->exp_val * pow(c->a->data[i], c->exp_val - 1.0);
        }
    }
}

TriadTensor *triad_tensor_pow(TriadTensor *a, double exp_val) {
    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    for (int64_t i = 0; i < a->size; i++) out->data[i] = pow(a->data[i], exp_val);
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        PowCtx *ctx = malloc(sizeof(PowCtx));
        ctx->a = a; ctx->exp_val = exp_val;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _pow_backward;
    }
    return out;
}

static void _matmul_backward(TriadTensor *out) {
    BinaryCtx *c = (BinaryCtx*)out->_ctx;
    int32_t M = c->a->shape[0], K = c->a->shape[1], N = c->b->shape[1];
    if (c->a->requires_grad) {
        _ensure_grad(c->a);

        for (int32_t i = 0; i < M; i++)
            for (int32_t j = 0; j < K; j++) {
                double s = 0;
                for (int32_t p = 0; p < N; p++)
                    s += out->grad[i * N + p] * c->b->data[j * N + p];
                c->a->grad[i * K + j] += s;
            }
    }
    if (c->b->requires_grad) {
        _ensure_grad(c->b);

        for (int32_t i = 0; i < K; i++)
            for (int32_t j = 0; j < N; j++) {
                double s = 0;
                for (int32_t p = 0; p < M; p++)
                    s += c->a->data[p * K + i] * out->grad[p * N + j];
                c->b->grad[i * N + j] += s;
            }
    }
}

TriadTensor *triad_tensor_matmul(TriadTensor *a, TriadTensor *b) {
    if (!a || !b || a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;
    int32_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
    int32_t shape[] = {M, N};
    TriadTensor *out = triad_tensor_new(2, shape, 0);

    #ifdef USE_CBLAS
    triad_matmul(M, K, N, a->data, b->data, out->data);
    #else
    for (int32_t i = 0; i < M; i++)
        for (int32_t j = 0; j < N; j++) {
            double s = 0;
            for (int32_t p = 0; p < K; p++)
                s += a->data[i * K + p] * b->data[p * N + j];
            out->data[i * N + j] = s;
        }
    #endif

    if (_triad_grad_enabled && (a->requires_grad || b->requires_grad)) {
        out->requires_grad = 1;
        _set_children(out, 2, a, b);
        BinaryCtx *ctx = malloc(sizeof(BinaryCtx));
        ctx->a = a; ctx->b = b; ctx->out = out;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _matmul_backward;
    }
    return out;
}

typedef struct { TriadTensor *a; } ViewCtx;

static void _reshape_backward(TriadTensor *out) {
    ViewCtx *c = (ViewCtx*)out->_ctx;
    if (c->a->requires_grad) _accumulate_grad(c->a, out->grad);
}

TriadTensor *triad_tensor_reshape(TriadTensor *a, int32_t ndim, const int32_t *shape) {
    if (_shape_size(ndim, shape) != a->size) return NULL;
    TriadTensor *out = triad_tensor_new(ndim, shape, 0);
    memcpy(out->data, a->data, (size_t)a->size * sizeof(double));
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        ViewCtx *ctx = malloc(sizeof(ViewCtx));
        ctx->a = a;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _reshape_backward;
    }
    return out;
}

TriadTensor *triad_tensor_flatten(TriadTensor *a) {
    int32_t shape[] = {(int32_t)a->size};
    return triad_tensor_reshape(a, 1, shape);
}

typedef struct { TriadTensor *a; int32_t axis1, axis2; } TransposeCtx;

static int64_t _flat_to_swapped_index(int64_t oi, const int32_t *out_shape,
                                      const int32_t *in_shape, int32_t ndim,
                                      int32_t axis1, int32_t axis2) {
    int32_t coords[32];
    int64_t tmp = oi;
    for (int32_t d = ndim - 1; d >= 0; d--) {
        coords[d] = (int32_t)(tmp % out_shape[d]);
        tmp /= out_shape[d];
    }
    int32_t ctmp = coords[axis1];
    coords[axis1] = coords[axis2];
    coords[axis2] = ctmp;

    int64_t idx = 0, stride = 1;
    for (int32_t d = ndim - 1; d >= 0; d--) {
        idx += coords[d] * stride;
        stride *= in_shape[d];
    }
    return idx;
}

static void _transpose_backward(TriadTensor *out) {
    TransposeCtx *c = (TransposeCtx*)out->_ctx;
    if (!c->a->requires_grad) return;
    _ensure_grad(c->a);
    for (int64_t oi = 0; oi < out->size; oi++) {
        int64_t ai = _flat_to_swapped_index(oi, out->shape, c->a->shape,
                                            out->ndim, c->axis1, c->axis2);
        c->a->grad[ai] += out->grad[oi];
    }
}

TriadTensor *triad_tensor_transpose(TriadTensor *a, int32_t axis1, int32_t axis2) {
    axis1 = _norm_axis(axis1, a->ndim);
    axis2 = _norm_axis(axis2, a->ndim);
    if (axis1 < 0 || axis1 >= a->ndim || axis2 < 0 || axis2 >= a->ndim) return NULL;
    int32_t shape[32];
    for (int32_t i = 0; i < a->ndim; i++) shape[i] = a->shape[i];
    int32_t tmp = shape[axis1];
    shape[axis1] = shape[axis2];
    shape[axis2] = tmp;
    TriadTensor *out = triad_tensor_new(a->ndim, shape, 0);
    for (int64_t oi = 0; oi < out->size; oi++) {
        int64_t ai = _flat_to_swapped_index(oi, out->shape, a->shape,
                                            out->ndim, axis1, axis2);
        out->data[oi] = a->data[ai];
    }
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        TransposeCtx *ctx = malloc(sizeof(TransposeCtx));
        ctx->a = a; ctx->axis1 = axis1; ctx->axis2 = axis2;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _transpose_backward;
    }
    return out;
}

static void _bmm_backward(TriadTensor *out) {
    BinaryCtx *c = (BinaryCtx*)out->_ctx;
    int32_t ndim = out->ndim;
    int32_t M = c->a->shape[ndim - 2];
    int32_t K = c->a->shape[ndim - 1];
    int32_t N = c->b->shape[ndim - 1];
    int64_t batches = out->size / ((int64_t)M * N);

    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        for (int64_t bt = 0; bt < batches; bt++) {
            double *ga = c->a->grad + bt * (int64_t)M * K;
            double *bb = c->b->data + bt * (int64_t)K * N;
            double *go = out->grad + bt * (int64_t)M * N;
            for (int32_t i = 0; i < M; i++)
                for (int32_t j = 0; j < K; j++) {
                    double s = 0.0;
                    for (int32_t p = 0; p < N; p++) s += go[i * N + p] * bb[j * N + p];
                    ga[i * K + j] += s;
                }
        }
    }
    if (c->b->requires_grad) {
        _ensure_grad(c->b);
        for (int64_t bt = 0; bt < batches; bt++) {
            double *aa = c->a->data + bt * (int64_t)M * K;
            double *gb = c->b->grad + bt * (int64_t)K * N;
            double *go = out->grad + bt * (int64_t)M * N;
            for (int32_t i = 0; i < K; i++)
                for (int32_t j = 0; j < N; j++) {
                    double s = 0.0;
                    for (int32_t p = 0; p < M; p++) s += aa[p * K + i] * go[p * N + j];
                    gb[i * N + j] += s;
                }
        }
    }
}

TriadTensor *triad_tensor_bmm(TriadTensor *a, TriadTensor *b) {
    if (a->ndim < 2 || b->ndim != a->ndim) return NULL;
    int32_t ndim = a->ndim;
    int32_t M = a->shape[ndim - 2], K = a->shape[ndim - 1];
    int32_t Kb = b->shape[ndim - 2], N = b->shape[ndim - 1];
    if (K != Kb) return NULL;
    int32_t shape[32];
    for (int32_t i = 0; i < ndim - 2; i++) {
        if (a->shape[i] != b->shape[i]) return NULL;
        shape[i] = a->shape[i];
    }
    shape[ndim - 2] = M;
    shape[ndim - 1] = N;
    TriadTensor *out = triad_tensor_new(ndim, shape, 0);
    int64_t batches = out->size / ((int64_t)M * N);
    for (int64_t bt = 0; bt < batches; bt++) {
        double *aa = a->data + bt * (int64_t)M * K;
        double *bb = b->data + bt * (int64_t)K * N;
        double *oo = out->data + bt * (int64_t)M * N;
        for (int32_t i = 0; i < M; i++)
            for (int32_t j = 0; j < N; j++) {
                double s = 0.0;
                for (int32_t p = 0; p < K; p++) s += aa[i * K + p] * bb[p * N + j];
                oo[i * N + j] = s;
            }
    }
    if (_triad_grad_enabled && (a->requires_grad || b->requires_grad)) {
        out->requires_grad = 1;
        _set_children(out, 2, a, b);
        BinaryCtx *ctx = malloc(sizeof(BinaryCtx));
        ctx->a = a; ctx->b = b; ctx->out = out;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _bmm_backward;
    }
    return out;
}

typedef struct {
    TriadTensor *a;
    int32_t axis;
    int32_t start;
    int32_t step;
} SliceCtx;

static int64_t _slice_input_index(TriadTensor *a, TriadTensor *out,
                                  int64_t oi, int32_t axis,
                                  int32_t start, int32_t step) {
    int32_t coords[32];
    int64_t tmp = oi;
    for (int32_t d = out->ndim - 1; d >= 0; d--) {
        coords[d] = (int32_t)(tmp % out->shape[d]);
        tmp /= out->shape[d];
    }
    coords[axis] = start + coords[axis] * step;
    int64_t ai = 0, stride = 1;
    for (int32_t d = a->ndim - 1; d >= 0; d--) {
        ai += coords[d] * stride;
        stride *= a->shape[d];
    }
    return ai;
}

static void _slice_backward(TriadTensor *out) {
    SliceCtx *c = (SliceCtx*)out->_ctx;
    if (!c->a->requires_grad) return;
    _ensure_grad(c->a);
    for (int64_t oi = 0; oi < out->size; oi++) {
        int64_t ai = _slice_input_index(c->a, out, oi, c->axis, c->start, c->step);
        c->a->grad[ai] += out->grad[oi];
    }
}

TriadTensor *triad_tensor_slice_axis(TriadTensor *a, int32_t axis,
                                     int32_t start, int32_t end, int32_t step) {
    if (!a || step == 0 || a->ndim <= 0 || a->ndim > 32) return NULL;
    axis = _norm_axis(axis, a->ndim);
    if (axis < 0 || axis >= a->ndim) return NULL;
    int32_t dim = a->shape[axis];

    if (start < 0) start += dim;
    if (end < 0) end += dim;
    if (step > 0) {
        if (start < 0) start = 0;
        if (start > dim) start = dim;
        if (end < 0) end = 0;
        if (end > dim) end = dim;
    } else {
        if (start < -1) start = -1;
        if (start >= dim) start = dim - 1;
        if (end < -1) end = -1;
        if (end >= dim) end = dim - 1;
    }

    int32_t len = 0;
    if (step > 0) {
        for (int32_t i = start; i < end; i += step) len++;
    } else {
        for (int32_t i = start; i > end; i += step) len++;
    }
    if (len < 0) len = 0;

    int32_t shape[32];
    for (int32_t i = 0; i < a->ndim; i++) shape[i] = a->shape[i];
    shape[axis] = len;
    TriadTensor *out = triad_tensor_new(a->ndim, shape, 0);
    for (int64_t oi = 0; oi < out->size; oi++) {
        int64_t ai = _slice_input_index(a, out, oi, axis, start, step);
        out->data[oi] = a->data[ai];
    }
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        SliceCtx *ctx = malloc(sizeof(SliceCtx));
        ctx->a = a; ctx->axis = axis; ctx->start = start; ctx->step = step;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _slice_backward;
    }
    return out;
}

typedef enum {
    CMP_EQ, CMP_NE, CMP_LT, CMP_LE, CMP_GT, CMP_GE
} CmpKind;

static int _cmp_eval(double x, double y, CmpKind kind) {
    switch (kind) {
        case CMP_EQ: return x == y;
        case CMP_NE: return x != y;
        case CMP_LT: return x < y;
        case CMP_LE: return x <= y;
        case CMP_GT: return x > y;
        case CMP_GE: return x >= y;
    }
    return 0;
}

static TriadTensor *_tensor_cmp(TriadTensor *a, TriadTensor *b, CmpKind kind) {
    if (!a || !b) return NULL;
    TriadTensor *out = NULL;
    if (a->size == 1 && b->size != 1) {
        out = triad_tensor_new(b->ndim, b->shape, 0);
        for (int64_t i = 0; i < b->size; i++)
            out->data[i] = _cmp_eval(a->data[0], b->data[i], kind) ? 1.0 : 0.0;
    } else if (b->size == 1) {
        out = triad_tensor_new(a->ndim, a->shape, 0);
        for (int64_t i = 0; i < a->size; i++)
            out->data[i] = _cmp_eval(a->data[i], b->data[0], kind) ? 1.0 : 0.0;
    } else {
        if (a->ndim != b->ndim || a->size != b->size) return NULL;
        for (int32_t d = 0; d < a->ndim; d++)
            if (a->shape[d] != b->shape[d]) return NULL;
        out = triad_tensor_new(a->ndim, a->shape, 0);
        for (int64_t i = 0; i < a->size; i++)
            out->data[i] = _cmp_eval(a->data[i], b->data[i], kind) ? 1.0 : 0.0;
    }
    return out;
}

TriadTensor *triad_tensor_eq(TriadTensor *a, TriadTensor *b) { return _tensor_cmp(a, b, CMP_EQ); }
TriadTensor *triad_tensor_ne(TriadTensor *a, TriadTensor *b) { return _tensor_cmp(a, b, CMP_NE); }
TriadTensor *triad_tensor_lt(TriadTensor *a, TriadTensor *b) { return _tensor_cmp(a, b, CMP_LT); }
TriadTensor *triad_tensor_le(TriadTensor *a, TriadTensor *b) { return _tensor_cmp(a, b, CMP_LE); }
TriadTensor *triad_tensor_gt(TriadTensor *a, TriadTensor *b) { return _tensor_cmp(a, b, CMP_GT); }
TriadTensor *triad_tensor_ge(TriadTensor *a, TriadTensor *b) { return _tensor_cmp(a, b, CMP_GE); }

typedef struct { TriadTensor *a; } SumCtx;

static void _sum_backward(TriadTensor *out) {
    SumCtx *c = (SumCtx*)out->_ctx;
    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        double g = out->grad[0];
        for (int64_t i = 0; i < c->a->size; i++)
            c->a->grad[i] += g;
    }
}

TriadTensor *triad_tensor_sum(TriadTensor *a) {
    double s = 0;
    for (int64_t i = 0; i < a->size; i++) s += a->data[i];
    TriadTensor *out = triad_tensor_scalar(s, 0);
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        SumCtx *ctx = malloc(sizeof(SumCtx));
        ctx->a = a;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _sum_backward;
    }
    return out;
}

static void _mean_backward(TriadTensor *out) {
    SumCtx *c = (SumCtx*)out->_ctx;
    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        double g = out->grad[0] / (double)c->a->size;
        for (int64_t i = 0; i < c->a->size; i++)
            c->a->grad[i] += g;
    }
}

TriadTensor *triad_tensor_mean(TriadTensor *a) {
    double s = 0;
    for (int64_t i = 0; i < a->size; i++) s += a->data[i];
    TriadTensor *out = triad_tensor_scalar(s / (double)a->size, 0);
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        SumCtx *ctx = malloc(sizeof(SumCtx));
        ctx->a = a;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _mean_backward;
    }
    return out;
}

typedef struct {
    TriadTensor *a;
    int32_t axis;
    int keepdims;
    int is_mean;
} AxisReduceCtx;

static int64_t _axis_out_index_from_input(TriadTensor *a, TriadTensor *out,
                                          int64_t ai, int32_t axis, int keepdims) {
    int32_t coords[32];
    int64_t tmp = ai;
    for (int32_t d = a->ndim - 1; d >= 0; d--) {
        coords[d] = (int32_t)(tmp % a->shape[d]);
        tmp /= a->shape[d];
    }

    int64_t oi = 0;
    int64_t stride = 1;
    if (keepdims) {
        for (int32_t d = out->ndim - 1; d >= 0; d--) {
            int32_t c = (d == axis) ? 0 : coords[d];
            oi += c * stride;
            stride *= out->shape[d];
        }
    } else {
        for (int32_t d = a->ndim - 1; d >= 0; d--) {
            if (d == axis) continue;
            oi += coords[d] * stride;
            int32_t od = d < axis ? d : d - 1;
            stride *= out->shape[od];
        }
    }
    return oi;
}

static void _axis_reduce_backward(TriadTensor *out) {
    AxisReduceCtx *c = (AxisReduceCtx*)out->_ctx;
    if (!c->a->requires_grad) return;
    _ensure_grad(c->a);
    double scale = c->is_mean ? 1.0 / (double)c->a->shape[c->axis] : 1.0;
    for (int64_t ai = 0; ai < c->a->size; ai++) {
        int64_t oi = _axis_out_index_from_input(c->a, out, ai, c->axis, c->keepdims);
        c->a->grad[ai] += out->grad[oi] * scale;
    }
}

static TriadTensor *_axis_reduce(TriadTensor *a, int32_t axis, int keepdims, int is_mean) {
    axis = _norm_axis(axis, a->ndim);
    if (axis < 0 || axis >= a->ndim) return NULL;
    int32_t ndim = keepdims ? a->ndim : a->ndim - 1;
    int32_t shape[32];
    if (keepdims) {
        for (int32_t i = 0; i < a->ndim; i++) shape[i] = (i == axis) ? 1 : a->shape[i];
    } else {
        for (int32_t i = 0, j = 0; i < a->ndim; i++) {
            if (i == axis) continue;
            shape[j++] = a->shape[i];
        }
    }
    TriadTensor *out = triad_tensor_new(ndim, shape, 0);
    for (int64_t ai = 0; ai < a->size; ai++) {
        int64_t oi = _axis_out_index_from_input(a, out, ai, axis, keepdims);
        out->data[oi] += a->data[ai];
    }
    if (is_mean) {
        double div = (double)a->shape[axis];
        for (int64_t i = 0; i < out->size; i++) out->data[i] /= div;
    }
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        AxisReduceCtx *ctx = malloc(sizeof(AxisReduceCtx));
        ctx->a = a; ctx->axis = axis; ctx->keepdims = keepdims; ctx->is_mean = is_mean;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _axis_reduce_backward;
    }
    return out;
}

TriadTensor *triad_tensor_sum_axis(TriadTensor *a, int32_t axis, int keepdims) {
    return _axis_reduce(a, axis, keepdims, 0);
}

TriadTensor *triad_tensor_mean_axis(TriadTensor *a, int32_t axis, int keepdims) {
    return _axis_reduce(a, axis, keepdims, 1);
}

typedef struct {
    TriadTensor *a;
    int32_t axis;
    int keepdims;
    int64_t *arg;
} AxisArgCtx;

static void _axis_arg_ctx_free(void *ptr) {
    AxisArgCtx *c = (AxisArgCtx*)ptr;
    if (!c) return;
    free(c->arg);
    free(c);
}

static void _axis_arg_backward(TriadTensor *out) {
    AxisArgCtx *c = (AxisArgCtx*)out->_ctx;
    if (!c->a->requires_grad) return;
    _ensure_grad(c->a);
    for (int64_t oi = 0; oi < out->size; oi++)
        c->a->grad[c->arg[oi]] += out->grad[oi];
}

static TriadTensor *_axis_minmax(TriadTensor *a, int32_t axis, int keepdims, int is_max) {
    axis = _norm_axis(axis, a->ndim);
    if (axis < 0 || axis >= a->ndim) return NULL;
    int32_t ndim = keepdims ? a->ndim : a->ndim - 1;
    int32_t shape[32];
    if (keepdims) {
        for (int32_t i = 0; i < a->ndim; i++) shape[i] = (i == axis) ? 1 : a->shape[i];
    } else {
        for (int32_t i = 0, j = 0; i < a->ndim; i++) {
            if (i == axis) continue;
            shape[j++] = a->shape[i];
        }
    }
    TriadTensor *out = triad_tensor_new(ndim, shape, 0);
    int64_t *arg = malloc((size_t)out->size * sizeof(int64_t));
    int *seen = calloc((size_t)out->size, sizeof(int));
    for (int64_t ai = 0; ai < a->size; ai++) {
        int64_t oi = _axis_out_index_from_input(a, out, ai, axis, keepdims);
        if (!seen[oi] ||
            (is_max ? a->data[ai] > out->data[oi] : a->data[ai] < out->data[oi])) {
            out->data[oi] = a->data[ai];
            arg[oi] = ai;
            seen[oi] = 1;
        }
    }
    free(seen);
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        AxisArgCtx *ctx = malloc(sizeof(AxisArgCtx));
        ctx->a = a; ctx->axis = axis; ctx->keepdims = keepdims; ctx->arg = arg;
        out->_ctx = ctx;
        out->ctx_free = _axis_arg_ctx_free;
        out->grad_fn = _axis_arg_backward;
    } else {
        free(arg);
    }
    return out;
}

TriadTensor *triad_tensor_max_axis(TriadTensor *a, int32_t axis, int keepdims) {
    return _axis_minmax(a, axis, keepdims, 1);
}

TriadTensor *triad_tensor_min_axis(TriadTensor *a, int32_t axis, int keepdims) {
    return _axis_minmax(a, axis, keepdims, 0);
}

typedef struct {
    TriadTensor **xs;
    int32_t n;
    int32_t axis;
    int is_stack;
} CatCtx;

static void _cat_ctx_free(void *ptr) {
    CatCtx *c = (CatCtx*)ptr;
    if (!c) return;
    free(c->xs);
    free(c);
}

static void _coords_from_flat(const TriadTensor *t, int64_t fi, int32_t *coords) {
    for (int32_t d = t->ndim - 1; d >= 0; d--) {
        coords[d] = (int32_t)(fi % t->shape[d]);
        fi /= t->shape[d];
    }
}

static void _cat_backward(TriadTensor *out) {
    CatCtx *c = (CatCtx*)out->_ctx;
    int32_t coords[32];

    int32_t starts[256];
    int32_t acc = 0;
    for (int32_t k = 0; k < c->n; k++) {
        starts[k] = acc;
        acc += c->is_stack ? 1 : c->xs[k]->shape[c->axis];
    }
    for (int64_t oi = 0; oi < out->size; oi++) {
        _coords_from_flat(out, oi, coords);
        int32_t a = coords[c->axis];

        int32_t k = 0;
        while (k + 1 < c->n && a >= starts[k + 1]) k++;
        TriadTensor *src = c->xs[k];
        if (!src->requires_grad) continue;
        _ensure_grad(src);

        int64_t si = 0, stride = 1;
        if (c->is_stack) {

            for (int32_t d = out->ndim - 1; d >= 0; d--) {
                if (d == c->axis) continue;
                int32_t sd = d < c->axis ? d : d - 1;
                si += coords[d] * stride;
                stride *= src->shape[sd];
            }
        } else {
            for (int32_t d = src->ndim - 1; d >= 0; d--) {
                int32_t cc = (d == c->axis) ? (a - starts[k]) : coords[d];
                si += cc * stride;
                stride *= src->shape[d];
            }
        }
        src->grad[si] += out->grad[oi];
    }
}

static TriadTensor *_cat_impl(TriadTensor **xs, int32_t n, int32_t axis, int is_stack) {
    if (n <= 0) return NULL;
    int32_t base_ndim = xs[0]->ndim;
    int32_t out_ndim = is_stack ? base_ndim + 1 : base_ndim;
    axis = _norm_axis(axis, out_ndim);
    if (axis < 0 || axis >= out_ndim) return NULL;

    int32_t shape[32];
    int any_grad = 0;
    if (is_stack) {

        for (int32_t d = 0, j = 0; d < out_ndim; d++) {
            if (d == axis) { shape[d] = n; continue; }
            shape[d] = xs[0]->shape[j++];
        }
    } else {
        for (int32_t d = 0; d < out_ndim; d++) shape[d] = xs[0]->shape[d];
        int32_t total = 0;
        for (int32_t k = 0; k < n; k++) total += xs[k]->shape[axis];
        shape[axis] = total;
    }
    for (int32_t k = 0; k < n; k++) if (xs[k]->requires_grad) any_grad = 1;

    TriadTensor *out = triad_tensor_new(out_ndim, shape, 0);

    int32_t coords[32];
    int32_t starts[256];
    int32_t acc = 0;
    for (int32_t k = 0; k < n; k++) {
        starts[k] = acc;
        acc += is_stack ? 1 : xs[k]->shape[axis];
    }
    for (int64_t oi = 0; oi < out->size; oi++) {
        _coords_from_flat(out, oi, coords);
        int32_t a = coords[axis];
        int32_t k = 0;
        while (k + 1 < n && a >= starts[k + 1]) k++;
        TriadTensor *src = xs[k];
        int64_t si = 0, stride = 1;
        if (is_stack) {
            for (int32_t d = out_ndim - 1; d >= 0; d--) {
                if (d == axis) continue;
                int32_t sd = d < axis ? d : d - 1;
                si += coords[d] * stride;
                stride *= src->shape[sd];
            }
        } else {
            for (int32_t d = src->ndim - 1; d >= 0; d--) {
                int32_t cc = (d == axis) ? (a - starts[k]) : coords[d];
                si += cc * stride;
                stride *= src->shape[d];
            }
        }
        out->data[oi] = src->data[si];
    }

    if (_triad_grad_enabled && any_grad) {
        out->requires_grad = 1;
        out->nchildren = n;
        out->children = malloc((size_t)n * sizeof(TriadTensor*));
        for (int32_t k = 0; k < n; k++) {
            out->children[k] = xs[k];
            triad_tensor_retain(xs[k]);
        }
        CatCtx *ctx = malloc(sizeof(CatCtx));
        ctx->xs = malloc((size_t)n * sizeof(TriadTensor*));
        memcpy(ctx->xs, xs, (size_t)n * sizeof(TriadTensor*));
        ctx->n = n; ctx->axis = axis; ctx->is_stack = is_stack;
        out->_ctx = ctx;
        out->ctx_free = _cat_ctx_free;
        out->grad_fn = _cat_backward;
    }
    return out;
}

TriadTensor *triad_tensor_cat(TriadTensor **xs, int32_t n, int32_t axis) {
    return _cat_impl(xs, n, axis, 0);
}

TriadTensor *triad_tensor_stack(TriadTensor **xs, int32_t n, int32_t axis) {
    return _cat_impl(xs, n, axis, 1);
}

typedef struct { TriadTensor *a; double *out_data; } MathCtx;

static void _exp_backward(TriadTensor *out) {
    MathCtx *c = (MathCtx*)out->_ctx;
    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        for (int64_t i = 0; i < c->a->size; i++)
            c->a->grad[i] += out->grad[i] * c->out_data[i];
    }
}

TriadTensor *triad_tensor_exp(TriadTensor *a) {
    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    for (int64_t i = 0; i < a->size; i++) out->data[i] = exp(a->data[i]);
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        MathCtx *ctx = malloc(sizeof(MathCtx));
        ctx->a = a; ctx->out_data = out->data;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _exp_backward;
    }
    return out;
}

static void _log_backward(TriadTensor *out) {
    MathCtx *c = (MathCtx*)out->_ctx;
    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        for (int64_t i = 0; i < c->a->size; i++)
            c->a->grad[i] += out->grad[i] / c->a->data[i];
    }
}

TriadTensor *triad_tensor_log(TriadTensor *a) {
    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    for (int64_t i = 0; i < a->size; i++) out->data[i] = log(a->data[i]);
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        MathCtx *ctx = malloc(sizeof(MathCtx));
        ctx->a = a; ctx->out_data = out->data;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _log_backward;
    }
    return out;
}

static void _sqrt_backward(TriadTensor *out) {
    MathCtx *c = (MathCtx*)out->_ctx;
    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        for (int64_t i = 0; i < c->a->size; i++)
            c->a->grad[i] += out->grad[i] * 0.5 / c->out_data[i];
    }
}

TriadTensor *triad_tensor_sqrt(TriadTensor *a) {
    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    for (int64_t i = 0; i < a->size; i++) out->data[i] = sqrt(a->data[i]);
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        MathCtx *ctx = malloc(sizeof(MathCtx));
        ctx->a = a; ctx->out_data = out->data;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _sqrt_backward;
    }
    return out;
}

static void _tanh_backward(TriadTensor *out) {
    MathCtx *c = (MathCtx*)out->_ctx;
    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        for (int64_t i = 0; i < c->a->size; i++)
            c->a->grad[i] += out->grad[i] * (1.0 - c->out_data[i] * c->out_data[i]);
    }
}

TriadTensor *triad_tensor_tanh(TriadTensor *a) {
    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    for (int64_t i = 0; i < a->size; i++) out->data[i] = tanh(a->data[i]);
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        MathCtx *ctx = malloc(sizeof(MathCtx));
        ctx->a = a; ctx->out_data = out->data;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _tanh_backward;
    }
    return out;
}

static void _sigmoid_backward(TriadTensor *out) {
    MathCtx *c = (MathCtx*)out->_ctx;
    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        for (int64_t i = 0; i < c->a->size; i++)
            c->a->grad[i] += out->grad[i] * c->out_data[i] * (1.0 - c->out_data[i]);
    }
}

TriadTensor *triad_tensor_sigmoid(TriadTensor *a) {
    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    for (int64_t i = 0; i < a->size; i++) out->data[i] = 1.0 / (1.0 + exp(-a->data[i]));
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        MathCtx *ctx = malloc(sizeof(MathCtx));
        ctx->a = a; ctx->out_data = out->data;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _sigmoid_backward;
    }
    return out;
}

static void _relu_backward(TriadTensor *out) {
    UnaryCtx *c = (UnaryCtx*)out->_ctx;
    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        for (int64_t i = 0; i < c->a->size; i++)
            c->a->grad[i] += out->grad[i] * (c->a->data[i] > 0 ? 1.0 : 0.0);
    }
}

TriadTensor *triad_tensor_relu(TriadTensor *a) {
    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    for (int64_t i = 0; i < a->size; i++) out->data[i] = a->data[i] > 0 ? a->data[i] : 0.0;
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        UnaryCtx *ctx = malloc(sizeof(UnaryCtx));
        ctx->a = a;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _relu_backward;
    }
    return out;
}

typedef struct { TriadTensor *a; double *mask; } DropoutCtx;

static void _dropout_ctx_free(void *ptr) {
    DropoutCtx *c = (DropoutCtx*)ptr;
    if (!c) return;
    free(c->mask);
    free(c);
}

static void _dropout_backward(TriadTensor *out) {
    DropoutCtx *c = (DropoutCtx*)out->_ctx;
    if (!c->a->requires_grad) return;
    _ensure_grad(c->a);
    for (int64_t i = 0; i < c->a->size; i++)
        c->a->grad[i] += out->grad[i] * c->mask[i];
}

TriadTensor *triad_tensor_dropout(TriadTensor *a, double p, int training) {
    if (!a || p < 0.0 || p >= 1.0) return NULL;
    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    if (!training || p == 0.0) {
        memcpy(out->data, a->data, (size_t)a->size * sizeof(double));
        if (_triad_grad_enabled && a->requires_grad) {
            out->requires_grad = 1;
            _set_children(out, 1, a);
            ViewCtx *ctx = malloc(sizeof(ViewCtx));
            ctx->a = a;
            out->_ctx = ctx;
            out->ctx_free = _ctx_free_plain;
            out->grad_fn = _reshape_backward;
        }
        return out;
    }

    double keep = 1.0 - p;
    double scale = 1.0 / keep;
    double *mask = malloc((size_t)a->size * sizeof(double));
    if (!mask) {
        triad_tensor_free(out);
        return NULL;
    }
    for (int64_t i = 0; i < a->size; i++) {
        mask[i] = (_randf() < keep) ? scale : 0.0;
        out->data[i] = a->data[i] * mask[i];
    }
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        DropoutCtx *ctx = malloc(sizeof(DropoutCtx));
        ctx->a = a;
        ctx->mask = mask;
        out->_ctx = ctx;
        out->ctx_free = _dropout_ctx_free;
        out->grad_fn = _dropout_backward;
    } else {
        free(mask);
    }
    return out;
}

typedef struct { TriadTensor *a; double *softmax_out; int32_t last_dim; } SoftmaxCtx;

static void _softmax_backward(TriadTensor *out) {
    SoftmaxCtx *c = (SoftmaxCtx*)out->_ctx;
    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        int32_t D = c->last_dim;
        int64_t batches = c->a->size / D;
        for (int64_t b = 0; b < batches; b++) {
            double dot = 0;
            for (int32_t j = 0; j < D; j++)
                dot += out->grad[b * D + j] * c->softmax_out[b * D + j];
            for (int32_t j = 0; j < D; j++)
                c->a->grad[b * D + j] += c->softmax_out[b * D + j] * (out->grad[b * D + j] - dot);
        }
    }
}

TriadTensor *triad_tensor_softmax(TriadTensor *a) {
    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    int32_t D = a->shape[a->ndim - 1];
    int64_t batches = a->size / D;
    for (int64_t b = 0; b < batches; b++) {
        double mx = -1e308;
        for (int32_t j = 0; j < D; j++)
            if (a->data[b * D + j] > mx) mx = a->data[b * D + j];
        double sum = 0;
        for (int32_t j = 0; j < D; j++) {
            out->data[b * D + j] = exp(a->data[b * D + j] - mx);
            sum += out->data[b * D + j];
        }
        for (int32_t j = 0; j < D; j++)
            out->data[b * D + j] /= sum;
    }
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        SoftmaxCtx *ctx = malloc(sizeof(SoftmaxCtx));
        ctx->a = a; ctx->softmax_out = out->data; ctx->last_dim = D;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _softmax_backward;
    }
    return out;
}

typedef struct {
    TriadTensor *a;
    double *softmax_out;
    int64_t outer;
    int32_t D;
    int64_t inner;
} SoftmaxAxisCtx;

static void _softmax_axis_backward(TriadTensor *out) {
    SoftmaxAxisCtx *c = (SoftmaxAxisCtx*)out->_ctx;
    if (!c->a->requires_grad) return;
    _ensure_grad(c->a);
    int32_t D = c->D;
    int64_t inner = c->inner;
    for (int64_t o = 0; o < c->outer; o++) {
        for (int64_t i = 0; i < inner; i++) {
            int64_t base = o * (int64_t)D * inner + i;
            double dot = 0;
            for (int32_t j = 0; j < D; j++) {
                int64_t idx = base + (int64_t)j * inner;
                dot += out->grad[idx] * c->softmax_out[idx];
            }
            for (int32_t j = 0; j < D; j++) {
                int64_t idx = base + (int64_t)j * inner;
                c->a->grad[idx] += c->softmax_out[idx] * (out->grad[idx] - dot);
            }
        }
    }
}

TriadTensor *triad_tensor_softmax_axis(TriadTensor *a, int32_t axis) {
    axis = _norm_axis(axis, a->ndim);
    if (axis < 0 || axis >= a->ndim) return NULL;
    int32_t D = a->shape[axis];
    int64_t inner = 1, outer = 1;
    for (int32_t d = axis + 1; d < a->ndim; d++) inner *= a->shape[d];
    for (int32_t d = 0; d < axis; d++) outer *= a->shape[d];

    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    for (int64_t o = 0; o < outer; o++) {
        for (int64_t i = 0; i < inner; i++) {
            int64_t base = o * (int64_t)D * inner + i;
            double mx = -1e308;
            for (int32_t j = 0; j < D; j++) {
                double v = a->data[base + (int64_t)j * inner];
                if (v > mx) mx = v;
            }
            double sum = 0;
            for (int32_t j = 0; j < D; j++) {
                int64_t idx = base + (int64_t)j * inner;
                out->data[idx] = exp(a->data[idx] - mx);
                sum += out->data[idx];
            }
            for (int32_t j = 0; j < D; j++)
                out->data[base + (int64_t)j * inner] /= sum;
        }
    }
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        SoftmaxAxisCtx *ctx = malloc(sizeof(SoftmaxAxisCtx));
        ctx->a = a; ctx->softmax_out = out->data;
        ctx->outer = outer; ctx->D = D; ctx->inner = inner;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _softmax_axis_backward;
    }
    return out;
}

typedef struct {
    TriadTensor *a, *gamma, *beta;
    double *x_hat;
    double *inv;
    int64_t rows;
    int32_t D;
} LayerNormCtx;

static void _layer_norm_ctx_free(void *ptr) {
    LayerNormCtx *c = (LayerNormCtx*)ptr;
    if (!c) return;
    free(c->x_hat);
    free(c->inv);
    free(c);
}

static void _layer_norm_backward(TriadTensor *out) {
    LayerNormCtx *c = (LayerNormCtx*)out->_ctx;
    int32_t D = c->D;

    if (c->beta && c->beta->requires_grad) {
        double *tmp = malloc((size_t)out->size * sizeof(double));
        memcpy(tmp, out->grad, (size_t)out->size * sizeof(double));
        double *ug = _unbroadcast(tmp, out->ndim, out->shape,
                                  c->beta->ndim, c->beta->shape, c->beta->size);
        _accumulate_grad(c->beta, ug);
        free(tmp);
        free(ug);
    }

    if (c->gamma && c->gamma->requires_grad) {
        double *tmp = malloc((size_t)out->size * sizeof(double));
        for (int64_t i = 0; i < out->size; i++) tmp[i] = out->grad[i] * c->x_hat[i];
        double *ug = _unbroadcast(tmp, out->ndim, out->shape,
                                  c->gamma->ndim, c->gamma->shape, c->gamma->size);
        _accumulate_grad(c->gamma, ug);
        free(tmp);
        free(ug);
    }

    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        for (int64_t r = 0; r < c->rows; r++) {
            double sum1 = 0.0, sum2 = 0.0;
            for (int32_t j = 0; j < D; j++) {
                int64_t idx = r * D + j;
                double gv = c->gamma ? c->gamma->data[j % c->gamma->size] : 1.0;
                double dxhat = out->grad[idx] * gv;
                sum1 += dxhat;
                sum2 += dxhat * c->x_hat[idx];
            }
            for (int32_t j = 0; j < D; j++) {
                int64_t idx = r * D + j;
                double gv = c->gamma ? c->gamma->data[j % c->gamma->size] : 1.0;
                double dxhat = out->grad[idx] * gv;
                c->a->grad[idx] += c->inv[r] / (double)D
                    * ((double)D * dxhat - sum1 - c->x_hat[idx] * sum2);
            }
        }
    }
}

TriadTensor *triad_tensor_layer_norm(TriadTensor *a, TriadTensor *gamma,
                                      TriadTensor *beta, double eps) {
    if (a->ndim < 1) return NULL;
    int32_t D = a->shape[a->ndim - 1];
    int64_t rows = a->size / D;
    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    double *x_hat = malloc((size_t)a->size * sizeof(double));
    double *inv = malloc((size_t)rows * sizeof(double));
    if (!x_hat || !inv) {
        free(x_hat);
        free(inv);
        triad_tensor_free(out);
        return NULL;
    }

    for (int64_t r = 0; r < rows; r++) {
        double mean = 0.0;
        for (int32_t j = 0; j < D; j++) mean += a->data[r * D + j];
        mean /= (double)D;
        double var = 0.0;
        for (int32_t j = 0; j < D; j++) {
            double d = a->data[r * D + j] - mean;
            var += d * d;
        }
        var /= (double)D;
        inv[r] = 1.0 / sqrt(var + eps);
        for (int32_t j = 0; j < D; j++) {
            int64_t idx = r * D + j;
            x_hat[idx] = (a->data[idx] - mean) * inv[r];
            double gv = gamma ? gamma->data[j % gamma->size] : 1.0;
            double bv = beta ? beta->data[j % beta->size] : 0.0;
            out->data[idx] = x_hat[idx] * gv + bv;
        }
    }

    int needs = a->requires_grad ||
        (gamma && gamma->requires_grad) ||
        (beta && beta->requires_grad);
    if (_triad_grad_enabled && needs) {
        out->requires_grad = 1;
        out->nchildren = 1 + (gamma ? 1 : 0) + (beta ? 1 : 0);
        out->children = malloc((size_t)out->nchildren * sizeof(TriadTensor*));
        int32_t ci = 0;
        out->children[ci++] = a; triad_tensor_retain(a);
        if (gamma) { out->children[ci++] = gamma; triad_tensor_retain(gamma); }
        if (beta) { out->children[ci++] = beta; triad_tensor_retain(beta); }

        LayerNormCtx *ctx = malloc(sizeof(LayerNormCtx));
        ctx->a = a; ctx->gamma = gamma; ctx->beta = beta;
        ctx->x_hat = x_hat; ctx->inv = inv; ctx->rows = rows; ctx->D = D;
        out->_ctx = ctx;
        out->ctx_free = _layer_norm_ctx_free;
        out->grad_fn = _layer_norm_backward;
    } else {
        free(x_hat);
        free(inv);
    }
    return out;
}

TriadTensor *triad_tensor_mse_loss(TriadTensor *pred, TriadTensor *target) {
    TriadTensor *diff = triad_tensor_sub(pred, target);
    TriadTensor *sq = triad_tensor_mul(diff, diff);
    return triad_tensor_mean(sq);
}

typedef struct { TriadTensor *logits; double *log_probs; int64_t n; int32_t V; int32_t *targets; } CECtx;

static void _ce_ctx_free(void *ptr) {
    CECtx *c = (CECtx*)ptr;
    if (!c) return;
    free(c->log_probs);
    free(c->targets);
    free(c);
}

static void _ce_backward(TriadTensor *out) {
    CECtx *c = (CECtx*)out->_ctx;
    if (c->logits->requires_grad) {
        _ensure_grad(c->logits);
        double g = out->grad[0];
        for (int64_t i = 0; i < c->n; i++) {
            for (int32_t j = 0; j < c->V; j++) {
                double p = exp(c->log_probs[i * c->V + j]);
                double grad = p;
                if (j == c->targets[i]) grad -= 1.0;
                c->logits->grad[i * c->V + j] += grad / (double)c->n * g;
            }
        }
    }
}

TriadTensor *triad_tensor_cross_entropy(TriadTensor *logits, TriadTensor *targets) {
    int32_t V = logits->shape[logits->ndim - 1];
    int64_t n = logits->size / V;

    double *log_probs = malloc(logits->size * sizeof(double));
    int32_t *tgt = malloc(n * sizeof(int32_t));

    for (int64_t i = 0; i < n; i++) {
        double mx = -1e308;
        for (int32_t j = 0; j < V; j++)
            if (logits->data[i * V + j] > mx) mx = logits->data[i * V + j];
        double lse = 0;
        for (int32_t j = 0; j < V; j++)
            lse += exp(logits->data[i * V + j] - mx);
        lse = mx + log(lse);
        for (int32_t j = 0; j < V; j++)
            log_probs[i * V + j] = logits->data[i * V + j] - lse;
        tgt[i] = (int32_t)targets->data[i];
    }

    double loss = 0;
    for (int64_t i = 0; i < n; i++)
        loss -= log_probs[i * V + tgt[i]];
    loss /= (double)n;

    TriadTensor *out = triad_tensor_scalar(loss, 0);
    if (_triad_grad_enabled && logits->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, logits);
        CECtx *ctx = malloc(sizeof(CECtx));
        ctx->logits = logits; ctx->log_probs = log_probs;
        ctx->n = n; ctx->V = V; ctx->targets = tgt;
        out->_ctx = ctx;
        out->ctx_free = _ce_ctx_free;
        out->grad_fn = _ce_backward;
    } else {
        free(log_probs);
        free(tgt);
    }
    return out;
}

typedef struct {
    TriadTensor *pred;
    TriadTensor *target;
    double delta;
    int kind;
} RegressionLossCtx;

static void _regression_loss_backward(TriadTensor *out) {
    RegressionLossCtx *c = (RegressionLossCtx*)out->_ctx;
    double g0 = out->grad[0] / (double)c->pred->size;
    if (c->pred->requires_grad) _ensure_grad(c->pred);
    if (c->target->requires_grad) _ensure_grad(c->target);
    for (int64_t i = 0; i < c->pred->size; i++) {
        double d = c->pred->data[i] - c->target->data[i];
        double gd;
        if (c->kind == 0) {
            gd = (d > 0.0) ? 1.0 : (d < 0.0 ? -1.0 : 0.0);
        } else {
            double ad = fabs(d);
            gd = (ad <= c->delta) ? d : c->delta * (d > 0.0 ? 1.0 : -1.0);
        }
        gd *= g0;
        if (c->pred->requires_grad) c->pred->grad[i] += gd;
        if (c->target->requires_grad) c->target->grad[i] -= gd;
    }
}

TriadTensor *triad_tensor_l1_loss(TriadTensor *pred, TriadTensor *target) {
    if (!pred || !target || pred->size != target->size) return NULL;
    double loss = 0.0;
    for (int64_t i = 0; i < pred->size; i++)
        loss += fabs(pred->data[i] - target->data[i]);
    loss /= (double)pred->size;
    TriadTensor *out = triad_tensor_scalar(loss, 0);
    if (_triad_grad_enabled && (pred->requires_grad || target->requires_grad)) {
        out->requires_grad = 1;
        _set_children(out, 2, pred, target);
        RegressionLossCtx *ctx = malloc(sizeof(RegressionLossCtx));
        ctx->pred = pred; ctx->target = target; ctx->delta = 1.0; ctx->kind = 0;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _regression_loss_backward;
    }
    return out;
}

TriadTensor *triad_tensor_huber_loss(TriadTensor *pred, TriadTensor *target, double delta) {
    if (!pred || !target || pred->size != target->size || delta <= 0.0) return NULL;
    double loss = 0.0;
    for (int64_t i = 0; i < pred->size; i++) {
        double d = pred->data[i] - target->data[i];
        double ad = fabs(d);
        loss += (ad <= delta) ? 0.5 * d * d : delta * (ad - 0.5 * delta);
    }
    loss /= (double)pred->size;
    TriadTensor *out = triad_tensor_scalar(loss, 0);
    if (_triad_grad_enabled && (pred->requires_grad || target->requires_grad)) {
        out->requires_grad = 1;
        _set_children(out, 2, pred, target);
        RegressionLossCtx *ctx = malloc(sizeof(RegressionLossCtx));
        ctx->pred = pred; ctx->target = target; ctx->delta = delta; ctx->kind = 1;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _regression_loss_backward;
    }
    return out;
}

TriadTensor *triad_tensor_smooth_l1_loss(TriadTensor *pred, TriadTensor *target) {
    return triad_tensor_huber_loss(pred, target, 1.0);
}

typedef struct {
    TriadTensor *x;
    TriadTensor *target;
    int from_logits;
} BCECtx;

static void _bce_backward(TriadTensor *out) {
    BCECtx *c = (BCECtx*)out->_ctx;
    double g0 = out->grad[0] / (double)c->x->size;
    if (c->x->requires_grad) _ensure_grad(c->x);
    if (c->target->requires_grad) _ensure_grad(c->target);
    const double eps = 1e-12;
    for (int64_t i = 0; i < c->x->size; i++) {
        double t = c->target->data[i];
        double gx, gt;
        if (c->from_logits) {
            double s = 1.0 / (1.0 + exp(-c->x->data[i]));
            gx = (s - t);
            gt = -c->x->data[i];
        } else {
            double p = c->x->data[i];
            if (p < eps) p = eps;
            if (p > 1.0 - eps) p = 1.0 - eps;
            gx = (p - t) / (p * (1.0 - p));
            gt = -(log(p) - log(1.0 - p));
        }
        if (c->x->requires_grad) c->x->grad[i] += gx * g0;
        if (c->target->requires_grad) c->target->grad[i] += gt * g0;
    }
}

TriadTensor *triad_tensor_bce_loss(TriadTensor *pred, TriadTensor *target) {
    if (!pred || !target || pred->size != target->size) return NULL;
    const double eps = 1e-12;
    double loss = 0.0;
    for (int64_t i = 0; i < pred->size; i++) {
        double p = pred->data[i], t = target->data[i];
        if (p < eps) p = eps;
        if (p > 1.0 - eps) p = 1.0 - eps;
        loss -= t * log(p) + (1.0 - t) * log(1.0 - p);
    }
    loss /= (double)pred->size;
    TriadTensor *out = triad_tensor_scalar(loss, 0);
    if (_triad_grad_enabled && (pred->requires_grad || target->requires_grad)) {
        out->requires_grad = 1;
        _set_children(out, 2, pred, target);
        BCECtx *ctx = malloc(sizeof(BCECtx));
        ctx->x = pred; ctx->target = target; ctx->from_logits = 0;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _bce_backward;
    }
    return out;
}

TriadTensor *triad_tensor_bce_with_logits(TriadTensor *logits, TriadTensor *target) {
    if (!logits || !target || logits->size != target->size) return NULL;

    double loss = 0.0;
    for (int64_t i = 0; i < logits->size; i++) {
        double x = logits->data[i], t = target->data[i];
        loss += (x > 0.0 ? x : 0.0) - x * t + log1p(exp(-fabs(x)));
    }
    loss /= (double)logits->size;
    TriadTensor *out = triad_tensor_scalar(loss, 0);
    if (_triad_grad_enabled && (logits->requires_grad || target->requires_grad)) {
        out->requires_grad = 1;
        _set_children(out, 2, logits, target);
        BCECtx *ctx = malloc(sizeof(BCECtx));
        ctx->x = logits; ctx->target = target; ctx->from_logits = 1;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _bce_backward;
    }
    return out;
}

typedef struct {
    TriadTensor *log_probs;
    int32_t *targets;
    int64_t n;
    int32_t V;
} NLLCtx;

static void _nll_ctx_free(void *ptr) {
    NLLCtx *c = (NLLCtx*)ptr;
    if (!c) return;
    free(c->targets);
    free(c);
}

static void _nll_backward(TriadTensor *out) {
    NLLCtx *c = (NLLCtx*)out->_ctx;
    if (!c->log_probs->requires_grad) return;
    _ensure_grad(c->log_probs);
    double g0 = out->grad[0] / (double)c->n;
    for (int64_t i = 0; i < c->n; i++) {
        int32_t t = c->targets[i];
        if (t < 0 || t >= c->V) continue;
        c->log_probs->grad[i * (int64_t)c->V + t] -= g0;
    }
}

TriadTensor *triad_tensor_nll_loss(TriadTensor *log_probs, TriadTensor *targets) {
    if (!log_probs || !targets || log_probs->ndim < 1) return NULL;
    int32_t V = log_probs->shape[log_probs->ndim - 1];
    int64_t n = log_probs->size / V;
    if (targets->size != n) return NULL;
    int32_t *tgt = malloc((size_t)n * sizeof(int32_t));
    double loss = 0.0;
    for (int64_t i = 0; i < n; i++) {
        int32_t t = (int32_t)targets->data[i];
        tgt[i] = t;
        if (t < 0 || t >= V) continue;
        loss -= log_probs->data[i * (int64_t)V + t];
    }
    loss /= (double)n;
    TriadTensor *out = triad_tensor_scalar(loss, 0);
    if (_triad_grad_enabled && log_probs->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, log_probs);
        NLLCtx *ctx = malloc(sizeof(NLLCtx));
        ctx->log_probs = log_probs; ctx->targets = tgt; ctx->n = n; ctx->V = V;
        out->_ctx = ctx;
        out->ctx_free = _nll_ctx_free;
        out->grad_fn = _nll_backward;
    } else {
        free(tgt);
    }
    return out;
}

typedef struct { TriadTensor *a; double scale; } ScaleCtx;

static void _scale_backward(TriadTensor *out) {
    ScaleCtx *c = (ScaleCtx*)out->_ctx;
    if (c->a->requires_grad) {
        _ensure_grad(c->a);
        for (int64_t i = 0; i < c->a->size; i++)
            c->a->grad[i] += out->grad[i] * c->scale;
    }
}

TriadTensor *triad_tensor_scale(TriadTensor *a, double s) {
    TriadTensor *out = triad_tensor_new(a->ndim, a->shape, 0);
    for (int64_t i = 0; i < a->size; i++) out->data[i] = a->data[i] * s;
    if (_triad_grad_enabled && a->requires_grad) {
        out->requires_grad = 1;
        _set_children(out, 1, a);
        ScaleCtx *ctx = malloc(sizeof(ScaleCtx));
        ctx->a = a; ctx->scale = s;
        out->_ctx = ctx;
        out->ctx_free = _ctx_free_plain;
        out->grad_fn = _scale_backward;
    }
    return out;
}
