#include "triad_ml.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void _ctx_free_plain(void *ctx) {
    free(ctx);
}

static uint64_t _nn_rng = 7654321ULL;

static double _nn_randf(void) {
    uint64_t x = _nn_rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    _nn_rng = x;
    return (x >> 11) * (1.0 / 9007199254740992.0);
}

typedef struct {
    TriadTensor  *x;
    TriadTensor  *weight;
    TriadTensor  *bias;
    int64_t       rows;
    int32_t       in_f;
    int32_t       out_f;
} triadCtx;

static void _triad_backward(TriadTensor *out) {
    triadCtx *c = (triadCtx*)out->_ctx;
    TriadTensor *inp = c->x;
    int64_t B = c->rows;
    int32_t inf = c->in_f, outf = c->out_f;

    if (inp->requires_grad) {
        if (!inp->grad) inp->grad = calloc(inp->size, sizeof(double));
        for (int64_t i = 0; i < B; i++)
            for (int32_t k = 0; k < inf; k++) {
                double s = 0;
                for (int32_t j = 0; j < outf; j++) {
                    int64_t gi = i * (int64_t)outf + j;
                    s += out->grad[gi] * c->weight->data[j * inf + k];
                }
                int64_t xi = i * (int64_t)inf + k;
                inp->grad[xi] += s;
            }
    }

    if (c->weight->requires_grad) {
        if (!c->weight->grad) c->weight->grad = calloc(c->weight->size, sizeof(double));
        for (int32_t j = 0; j < outf; j++)
            for (int32_t k = 0; k < inf; k++) {
                double s = 0;
                for (int64_t i = 0; i < B; i++) {
                    int64_t gi = i * (int64_t)outf + j;
                    int64_t xi = i * (int64_t)inf + k;
                    s += out->grad[gi] * inp->data[xi];
                }
                c->weight->grad[j * inf + k] += s;
            }
    }

    if (c->bias && c->bias->requires_grad) {
        if (!c->bias->grad) c->bias->grad = calloc(c->bias->size, sizeof(double));
        for (int32_t j = 0; j < outf; j++) {
            double s = 0;
            for (int64_t i = 0; i < B; i++) {
                int64_t gi = i * (int64_t)outf + j;
                s += out->grad[gi];
            }
            c->bias->grad[j] += s;
        }
    }
}

Triadtriad *triad_triad_new(int32_t in_f, int32_t out_f, int use_bias) {
    Triadtriad *l = malloc(sizeof(Triadtriad));
    l->in_features = in_f;
    l->out_features = out_f;

    int32_t wshape[] = {out_f, in_f};
    l->weight = triad_tensor_new(2, wshape, 1);
    double bound = sqrt(1.0 / (double)in_f);
    for (int64_t i = 0; i < (int64_t)out_f * in_f; i++)
        l->weight->data[i] = (_nn_randf() * 2.0 - 1.0) * bound;

    if (use_bias) {
        int32_t bshape[] = {out_f};
        l->bias = triad_tensor_new(1, bshape, 1);
        for (int32_t i = 0; i < out_f; i++)
            l->bias->data[i] = (_nn_randf() * 2.0 - 1.0) * bound;
    } else {
        l->bias = NULL;
    }
    return l;
}

void triad_triad_free(Triadtriad *l) {
    if (!l) return;
    triad_tensor_free(l->weight);
    if (l->bias) triad_tensor_free(l->bias);
    free(l);
}

TriadTensor *triad_triad_forward(Triadtriad *l, TriadTensor *x) {
    int32_t in_f = l->in_features, out_f = l->out_features;
    if (x->ndim < 1 || x->shape[x->ndim - 1] != in_f) return NULL;
    int64_t rows = x->size / in_f;

    int32_t oshape[32];
    for (int32_t i = 0; i < x->ndim - 1; i++) oshape[i] = x->shape[i];
    oshape[x->ndim - 1] = out_f;
    TriadTensor *result = triad_tensor_new(x->ndim, oshape, 0);

    for (int64_t i = 0; i < rows; i++) {
        for (int32_t j = 0; j < out_f; j++) {
            double s = 0;
            for (int32_t k = 0; k < in_f; k++) {
                int64_t xi = i * (int64_t)in_f + k;
                s += x->data[xi] * l->weight->data[j * in_f + k];
            }
            if (l->bias) s += l->bias->data[j];
            int64_t oi = i * (int64_t)out_f + j;
            result->data[oi] = s;
        }
    }

    int needs = x->requires_grad || l->weight->requires_grad ||
                (l->bias && l->bias->requires_grad);
    if (needs) {
        result->requires_grad = 1;

        int nc = 2 + (l->bias ? 1 : 0);
        result->nchildren = nc;
        result->children = malloc(nc * sizeof(TriadTensor*));
        result->children[0] = x; triad_tensor_retain(x);
        result->children[1] = l->weight; triad_tensor_retain(l->weight);
        if (l->bias) { result->children[2] = l->bias; triad_tensor_retain(l->bias); }

        triadCtx *ctx = malloc(sizeof(triadCtx));
        ctx->x = x;
        ctx->weight = l->weight;
        ctx->bias = l->bias;
        ctx->rows = rows;
        ctx->in_f = in_f;
        ctx->out_f = out_f;
        result->_ctx = ctx;
        result->ctx_free = _ctx_free_plain;
        result->grad_fn = _triad_backward;
    }

    return result;
}

typedef struct {
    TriadTensor *weight;
    int32_t *ids;
    int64_t nids;
    int32_t embedding_dim;
} EmbeddingCtx;

static void _embedding_ctx_free(void *ptr) {
    EmbeddingCtx *c = (EmbeddingCtx*)ptr;
    if (!c) return;
    free(c->ids);
    free(c);
}

static void _embedding_backward(TriadTensor *out) {
    EmbeddingCtx *c = (EmbeddingCtx*)out->_ctx;
    if (!c->weight->requires_grad) return;
    if (!c->weight->grad) c->weight->grad = calloc(c->weight->size, sizeof(double));
    int32_t D = c->embedding_dim;
    for (int64_t i = 0; i < c->nids; i++) {
        int32_t id = c->ids[i];
        if (id < 0 || id >= c->weight->shape[0]) continue;
        for (int32_t j = 0; j < D; j++)
            c->weight->grad[(int64_t)id * D + j] += out->grad[i * (int64_t)D + j];
    }
}

TriadEmbedding *triad_embedding_new(int32_t num_embeddings, int32_t embedding_dim) {
    TriadEmbedding *e = malloc(sizeof(TriadEmbedding));
    e->num_embeddings = num_embeddings;
    e->embedding_dim = embedding_dim;
    int32_t shape[] = {num_embeddings, embedding_dim};
    e->weight = triad_tensor_new(2, shape, 1);
    for (int64_t i = 0; i < e->weight->size; i++)
        e->weight->data[i] = (_nn_randf() * 2.0 - 1.0) * 0.02;
    return e;
}

void triad_embedding_free(TriadEmbedding *e) {
    if (!e) return;
    triad_tensor_free(e->weight);
    free(e);
}

TriadTensor *triad_embedding_forward(TriadEmbedding *e, TriadTensor *idx) {
    int32_t oshape[32];
    for (int32_t i = 0; i < idx->ndim; i++) oshape[i] = idx->shape[i];
    oshape[idx->ndim] = e->embedding_dim;
    TriadTensor *out = triad_tensor_new(idx->ndim + 1, oshape, 0);
    int32_t *ids = malloc((size_t)idx->size * sizeof(int32_t));
    for (int64_t i = 0; i < idx->size; i++) {
        int32_t id = (int32_t)idx->data[i];
        ids[i] = id;
        if (id < 0 || id >= e->num_embeddings) continue;
        memcpy(out->data + i * (int64_t)e->embedding_dim,
               e->weight->data + id * (int64_t)e->embedding_dim,
               (size_t)e->embedding_dim * sizeof(double));
    }
    if (e->weight->requires_grad) {
        out->requires_grad = 1;
        out->nchildren = 1;
        out->children = malloc(sizeof(TriadTensor*));
        out->children[0] = e->weight;
        triad_tensor_retain(e->weight);
        EmbeddingCtx *ctx = malloc(sizeof(EmbeddingCtx));
        ctx->weight = e->weight;
        ctx->ids = ids;
        ctx->nids = idx->size;
        ctx->embedding_dim = e->embedding_dim;
        out->_ctx = ctx;
        out->ctx_free = _embedding_ctx_free;
        out->grad_fn = _embedding_backward;
    } else {
        free(ids);
    }
    return out;
}

TriadLayerNorm *triad_layer_norm_new(int32_t normalized_shape, double eps) {
    TriadLayerNorm *ln = malloc(sizeof(TriadLayerNorm));
    ln->normalized_shape = normalized_shape;
    ln->eps = eps;
    int32_t shape[] = {normalized_shape};
    ln->gamma = triad_tensor_ones(1, shape, 1);
    ln->beta = triad_tensor_zeros(1, shape, 1);
    return ln;
}

void triad_layer_norm_free(TriadLayerNorm *ln) {
    if (!ln) return;
    triad_tensor_free(ln->gamma);
    triad_tensor_free(ln->beta);
    free(ln);
}

TriadTensor *triad_layer_norm_forward(TriadLayerNorm *ln, TriadTensor *x) {
    if (x->ndim < 1 || x->shape[x->ndim - 1] != ln->normalized_shape) return NULL;
    return triad_tensor_layer_norm(x, ln->gamma, ln->beta, ln->eps);
}

TriadBatchNorm1d *triad_batch_norm1d_new(int32_t num_features, double eps, double momentum) {
    TriadBatchNorm1d *bn = calloc(1, sizeof(TriadBatchNorm1d));
    if (!bn) return NULL;
    bn->num_features = num_features;
    bn->eps = eps > 0.0 ? eps : 1e-5;
    bn->momentum = momentum > 0.0 ? momentum : 0.1;
    bn->training = 1;
    int32_t shape[] = {num_features};
    bn->gamma = triad_tensor_ones(1, shape, 1);
    bn->beta = triad_tensor_zeros(1, shape, 1);
    bn->running_mean = calloc((size_t)num_features, sizeof(double));
    bn->running_var = malloc((size_t)num_features * sizeof(double));
    if (!bn->gamma || !bn->beta || !bn->running_mean || !bn->running_var) {
        triad_batch_norm1d_free(bn);
        return NULL;
    }
    for (int32_t i = 0; i < num_features; i++) bn->running_var[i] = 1.0;
    return bn;
}

void triad_batch_norm1d_free(TriadBatchNorm1d *bn) {
    if (!bn) return;
    triad_tensor_free(bn->gamma);
    triad_tensor_free(bn->beta);
    free(bn->running_mean);
    free(bn->running_var);
    free(bn);
}

void triad_batch_norm1d_train(TriadBatchNorm1d *bn, int training) {
    if (bn) bn->training = training;
}

TriadTensor *triad_batch_norm1d_forward(TriadBatchNorm1d *bn, TriadTensor *x) {
    if (!bn || !x || x->ndim != 2 || x->shape[1] != bn->num_features) return NULL;
    int32_t keep_shape[] = {1, bn->num_features};

    TriadTensor *mean = NULL;
    TriadTensor *var = NULL;
    if (bn->training) {
        mean = triad_tensor_mean_axis(x, 0, 1);
        TriadTensor *centered_tmp = triad_tensor_sub(x, mean);
        TriadTensor *sq = triad_tensor_mul(centered_tmp, centered_tmp);
        var = triad_tensor_mean_axis(sq, 0, 1);
        for (int32_t j = 0; j < bn->num_features; j++) {
            bn->running_mean[j] = (1.0 - bn->momentum) * bn->running_mean[j]
                                + bn->momentum * mean->data[j];
            bn->running_var[j] = (1.0 - bn->momentum) * bn->running_var[j]
                               + bn->momentum * var->data[j];
        }
        triad_tensor_free(centered_tmp);
        triad_tensor_free(sq);
    } else {
        mean = triad_tensor_from_data(2, keep_shape, bn->running_mean, 0);
        var = triad_tensor_from_data(2, keep_shape, bn->running_var, 0);
    }

    TriadTensor *centered = triad_tensor_sub(x, mean);
    TriadTensor *eps = triad_tensor_scalar(bn->eps, 0);
    TriadTensor *var_eps = triad_tensor_add(var, eps);
    TriadTensor *std = triad_tensor_sqrt(var_eps);
    TriadTensor *norm = triad_tensor_div(centered, std);
    TriadTensor *scaled = triad_tensor_mul(norm, bn->gamma);
    TriadTensor *out = triad_tensor_add(scaled, bn->beta);

    triad_tensor_free(mean);
    triad_tensor_free(var);
    triad_tensor_free(centered);
    triad_tensor_free(eps);
    triad_tensor_free(var_eps);
    triad_tensor_free(std);
    triad_tensor_free(norm);
    triad_tensor_free(scaled);
    return out;
}

TriadSequential *triad_sequential_new(int32_t nlayers) {
    TriadSequential *s = malloc(sizeof(TriadSequential));
    s->nlayers = nlayers;
    s->layers = calloc(nlayers, sizeof(TriadLayerEntry));
    return s;
}

void triad_sequential_set(TriadSequential *s, int32_t i,
                           TriadLayerType type, void *layer) {
    s->layers[i].type = type;
    s->layers[i].layer = layer;
}

void triad_sequential_free(TriadSequential *s) {
    if (!s) return;
    for (int32_t i = 0; i < s->nlayers; i++) {
        if (s->layers[i].type == TRIAD_LAYER_triad)
            triad_triad_free(s->layers[i].layer);
        else if (s->layers[i].type == TRIAD_LAYER_EMBEDDING)
            triad_embedding_free(s->layers[i].layer);
        else if (s->layers[i].type == TRIAD_LAYER_LAYER_NORM)
            triad_layer_norm_free(s->layers[i].layer);
    }
    free(s->layers);
    free(s);
}

TriadTensor *triad_sequential_forward(TriadSequential *s, TriadTensor *x) {
    TriadTensor *h = x;
    for (int32_t i = 0; i < s->nlayers; i++) {
        switch (s->layers[i].type) {
            case TRIAD_LAYER_triad:
                h = triad_triad_forward((Triadtriad*)s->layers[i].layer, h);
                break;
            case TRIAD_LAYER_RELU:
                h = triad_tensor_relu(h);
                break;
            case TRIAD_LAYER_SIGMOID:
                h = triad_tensor_sigmoid(h);
                break;
            case TRIAD_LAYER_TANH:
                h = triad_tensor_tanh(h);
                break;
            case TRIAD_LAYER_SOFTMAX:
                h = triad_tensor_softmax(h);
                break;
            case TRIAD_LAYER_FLATTEN:
                if (h->ndim <= 1) break;
                {
                    int32_t shape[] = {h->shape[0], (int32_t)(h->size / h->shape[0])};
                    h = triad_tensor_reshape(h, 2, shape);
                }
                break;
            case TRIAD_LAYER_EMBEDDING:
                h = triad_embedding_forward((TriadEmbedding*)s->layers[i].layer, h);
                break;
            case TRIAD_LAYER_LAYER_NORM:
                h = triad_layer_norm_forward((TriadLayerNorm*)s->layers[i].layer, h);
                break;
        }
        if (!h) return NULL;
    }
    return h;
}

int32_t triad_sequential_params(TriadSequential *s, TriadTensor **out, int32_t max) {
    int32_t n = 0;
    for (int32_t i = 0; i < s->nlayers && n < max; i++) {
        if (s->layers[i].type == TRIAD_LAYER_triad) {
            Triadtriad *l = (Triadtriad*)s->layers[i].layer;
            out[n++] = l->weight;
            if (l->bias && n < max) out[n++] = l->bias;
        } else if (s->layers[i].type == TRIAD_LAYER_EMBEDDING) {
            TriadEmbedding *e = (TriadEmbedding*)s->layers[i].layer;
            out[n++] = e->weight;
        } else if (s->layers[i].type == TRIAD_LAYER_LAYER_NORM) {
            TriadLayerNorm *ln = (TriadLayerNorm*)s->layers[i].layer;
            out[n++] = ln->gamma;
            if (n < max) out[n++] = ln->beta;
        }
    }
    return n;
}

TriadSGD *triad_sgd_new(TriadTensor **params, int32_t n, double lr, double momentum) {
    TriadSGD *opt = malloc(sizeof(TriadSGD));
    opt->nparams = n;
    opt->params = malloc(n * sizeof(TriadTensor*));
    memcpy(opt->params, params, n * sizeof(TriadTensor*));
    opt->lr = lr;
    opt->momentum = momentum;
    if (momentum > 0) {
        opt->velocity = malloc(n * sizeof(double*));
        for (int32_t i = 0; i < n; i++)
            opt->velocity[i] = calloc(params[i]->size, sizeof(double));
    } else {
        opt->velocity = NULL;
    }
    return opt;
}

void triad_sgd_step(TriadSGD *opt) {
    for (int32_t i = 0; i < opt->nparams; i++) {
        TriadTensor *p = opt->params[i];
        if (!p->grad) continue;
        if (opt->momentum > 0) {
            for (int64_t j = 0; j < p->size; j++) {
                opt->velocity[i][j] = opt->momentum * opt->velocity[i][j] + p->grad[j];
                p->data[j] -= opt->lr * opt->velocity[i][j];
            }
        } else {
            for (int64_t j = 0; j < p->size; j++)
                p->data[j] -= opt->lr * p->grad[j];
        }
    }
}

void triad_sgd_zero_grad(TriadSGD *opt) {
    for (int32_t i = 0; i < opt->nparams; i++)
        triad_tensor_zero_grad(opt->params[i]);
}

void triad_sgd_free(TriadSGD *opt) {
    if (!opt) return;
    if (opt->velocity) {
        for (int32_t i = 0; i < opt->nparams; i++) free(opt->velocity[i]);
        free(opt->velocity);
    }
    free(opt->params);
    free(opt);
}

TriadAdam *triad_adam_new(TriadTensor **params, int32_t n,
                           double lr, double beta1, double beta2, double eps) {
    TriadAdam *opt = malloc(sizeof(TriadAdam));
    opt->nparams = n;
    opt->params = malloc(n * sizeof(TriadTensor*));
    memcpy(opt->params, params, n * sizeof(TriadTensor*));
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->t = 0;
    opt->m = malloc(n * sizeof(double*));
    opt->v = malloc(n * sizeof(double*));
    for (int32_t i = 0; i < n; i++) {
        opt->m[i] = calloc(params[i]->size, sizeof(double));
        opt->v[i] = calloc(params[i]->size, sizeof(double));
    }
    return opt;
}

void triad_adam_step(TriadAdam *opt) {
    opt->t++;
    double bc1 = 1.0 - pow(opt->beta1, opt->t);
    double bc2 = 1.0 - pow(opt->beta2, opt->t);
    for (int32_t i = 0; i < opt->nparams; i++) {
        TriadTensor *p = opt->params[i];
        if (!p->grad) continue;
        for (int64_t j = 0; j < p->size; j++) {
            double g = p->grad[j];
            opt->m[i][j] = opt->beta1 * opt->m[i][j] + (1.0 - opt->beta1) * g;
            opt->v[i][j] = opt->beta2 * opt->v[i][j] + (1.0 - opt->beta2) * g * g;
            double m_hat = opt->m[i][j] / bc1;
            double v_hat = opt->v[i][j] / bc2;
            p->data[j] -= opt->lr * m_hat / (sqrt(v_hat) + opt->eps);
        }
    }
}

void triad_adam_zero_grad(TriadAdam *opt) {
    for (int32_t i = 0; i < opt->nparams; i++)
        triad_tensor_zero_grad(opt->params[i]);
}

void triad_adam_free(TriadAdam *opt) {
    if (!opt) return;
    for (int32_t i = 0; i < opt->nparams; i++) {
        free(opt->m[i]);
        free(opt->v[i]);
    }
    free(opt->m);
    free(opt->v);
    free(opt->params);
    free(opt);
}

TriadFeedForward *triad_feedforward_new(int32_t d_model, int32_t d_ff) {
    TriadFeedForward *ff = malloc(sizeof(TriadFeedForward));
    ff->d_model = d_model;
    ff->d_ff = d_ff;
    ff->fc1 = triad_triad_new(d_model, d_ff, 1);
    ff->fc2 = triad_triad_new(d_ff, d_model, 1);
    return ff;
}

void triad_feedforward_free(TriadFeedForward *ff) {
    if (!ff) return;
    triad_triad_free(ff->fc1);
    triad_triad_free(ff->fc2);
    free(ff);
}

TriadTensor *triad_feedforward_forward(TriadFeedForward *ff, TriadTensor *x) {
    TriadTensor *h = triad_triad_forward(ff->fc1, x);
    if (!h) return NULL;
    TriadTensor *a = triad_tensor_relu(h);
    TriadTensor *out = triad_triad_forward(ff->fc2, a);
    return out;
}

int32_t triad_feedforward_params(TriadFeedForward *ff, TriadTensor **out, int32_t max) {
    int32_t n = 0;
    if (n < max) out[n++] = ff->fc1->weight;
    if (ff->fc1->bias && n < max) out[n++] = ff->fc1->bias;
    if (n < max) out[n++] = ff->fc2->weight;
    if (ff->fc2->bias && n < max) out[n++] = ff->fc2->bias;
    return n;
}

TriadMultiHeadAttention *triad_mha_new(int32_t d_model, int32_t n_heads) {
    TriadMultiHeadAttention *m = malloc(sizeof(TriadMultiHeadAttention));
    m->d_model = d_model;
    m->n_heads = n_heads;
    m->q_proj = triad_triad_new(d_model, d_model, 1);
    m->k_proj = triad_triad_new(d_model, d_model, 1);
    m->v_proj = triad_triad_new(d_model, d_model, 1);
    m->out_proj = triad_triad_new(d_model, d_model, 1);
    return m;
}

void triad_mha_free(TriadMultiHeadAttention *m) {
    if (!m) return;
    triad_triad_free(m->q_proj);
    triad_triad_free(m->k_proj);
    triad_triad_free(m->v_proj);
    triad_triad_free(m->out_proj);
    free(m);
}

static TriadTensor *_split_heads(TriadTensor *x, int32_t B, int32_t T,
                                 int32_t H, int32_t dh) {
    int32_t s4[] = {B, T, H, dh};
    TriadTensor *r = triad_tensor_reshape(x, 4, s4);
    return triad_tensor_transpose(r, 1, 2);
}

TriadTensor *triad_mha_forward(TriadMultiHeadAttention *m, TriadTensor *x) {

    int wrapped = 0;
    int32_t B, T, d = m->d_model;
    if (x->ndim == 2) {
        B = 1; T = x->shape[0];
        int32_t s3[] = {1, T, d};
        x = triad_tensor_reshape(x, 3, s3);
        wrapped = 1;
    } else if (x->ndim == 3) {
        B = x->shape[0]; T = x->shape[1];
    } else {
        return NULL;
    }
    if (x->shape[x->ndim - 1] != d) return NULL;
    int32_t H = m->n_heads, dh = d / H;

    TriadTensor *q = triad_triad_forward(m->q_proj, x);
    TriadTensor *k = triad_triad_forward(m->k_proj, x);
    TriadTensor *v = triad_triad_forward(m->v_proj, x);

    TriadTensor *qh = _split_heads(q, B, T, H, dh);
    TriadTensor *kh = _split_heads(k, B, T, H, dh);
    TriadTensor *vh = _split_heads(v, B, T, H, dh);

    TriadTensor *kt = triad_tensor_transpose(kh, 2, 3);
    TriadTensor *scores = triad_tensor_bmm(qh, kt);
    TriadTensor *scaled = triad_tensor_scale(scores, 1.0 / sqrt((double)dh));
    TriadTensor *attn = triad_tensor_softmax_axis(scaled, 3);
    TriadTensor *ctx = triad_tensor_bmm(attn, vh);

    TriadTensor *ctxt = triad_tensor_transpose(ctx, 1, 2);
    int32_t s3[] = {B, T, d};
    TriadTensor *merged = triad_tensor_reshape(ctxt, 3, s3);
    TriadTensor *out = triad_triad_forward(m->out_proj, merged);

    if (wrapped && out) {
        int32_t s2[] = {T, d};
        out = triad_tensor_reshape(out, 2, s2);
    }
    return out;
}

int32_t triad_mha_params(TriadMultiHeadAttention *m, TriadTensor **out, int32_t max) {
    Triadtriad *ls[] = {m->q_proj, m->k_proj, m->v_proj, m->out_proj};
    int32_t n = 0;
    for (int i = 0; i < 4; i++) {
        if (n < max) out[n++] = ls[i]->weight;
        if (ls[i]->bias && n < max) out[n++] = ls[i]->bias;
    }
    return n;
}

TriadTransformerBlock *triad_transformer_block_new(int32_t d_model, int32_t n_heads,
                                                   int32_t d_ff) {
    TriadTransformerBlock *b = malloc(sizeof(TriadTransformerBlock));
    b->d_model = d_model;
    b->ln1 = triad_layer_norm_new(d_model, 1e-5);
    b->attn = triad_mha_new(d_model, n_heads);
    b->ln2 = triad_layer_norm_new(d_model, 1e-5);
    b->ff = triad_feedforward_new(d_model, d_ff);
    return b;
}

void triad_transformer_block_free(TriadTransformerBlock *b) {
    if (!b) return;
    triad_layer_norm_free(b->ln1);
    triad_mha_free(b->attn);
    triad_layer_norm_free(b->ln2);
    triad_feedforward_free(b->ff);
    free(b);
}

TriadTensor *triad_transformer_block_forward(TriadTransformerBlock *b, TriadTensor *x) {

    TriadTensor *n1 = triad_layer_norm_forward(b->ln1, x);
    if (!n1) return NULL;
    TriadTensor *a = triad_mha_forward(b->attn, n1);
    if (!a) return NULL;
    TriadTensor *h = triad_tensor_add(x, a);

    TriadTensor *n2 = triad_layer_norm_forward(b->ln2, h);
    TriadTensor *f = triad_feedforward_forward(b->ff, n2);
    TriadTensor *out = triad_tensor_add(h, f);
    return out;
}

int32_t triad_transformer_block_params(TriadTransformerBlock *b, TriadTensor **out,
                                       int32_t max) {
    int32_t n = 0;
    if (n < max) out[n++] = b->ln1->gamma;
    if (n < max) out[n++] = b->ln1->beta;
    n += triad_mha_params(b->attn, out + n, max - n);
    if (n < max) out[n++] = b->ln2->gamma;
    if (n < max) out[n++] = b->ln2->beta;
    n += triad_feedforward_params(b->ff, out + n, max - n);
    return n;
}

TriadTransformer *triad_transformer_new(int32_t vocab, int32_t d_model,
                                        int32_t n_blocks, int32_t n_heads, int32_t d_ff) {
    TriadTransformer *t = malloc(sizeof(TriadTransformer));
    t->d_model = d_model;
    t->n_blocks = n_blocks;
    t->embed = triad_embedding_new(vocab, d_model);
    t->blocks = malloc((size_t)n_blocks * sizeof(TriadTransformerBlock*));
    for (int32_t i = 0; i < n_blocks; i++)
        t->blocks[i] = triad_transformer_block_new(d_model, n_heads, d_ff);
    t->ln_final = triad_layer_norm_new(d_model, 1e-5);
    return t;
}

void triad_transformer_free(TriadTransformer *t) {
    if (!t) return;
    triad_embedding_free(t->embed);
    for (int32_t i = 0; i < t->n_blocks; i++)
        triad_transformer_block_free(t->blocks[i]);
    free(t->blocks);
    triad_layer_norm_free(t->ln_final);
    free(t);
}

TriadTensor *triad_transformer_forward(TriadTransformer *t, TriadTensor *idx) {
    TriadTensor *h = triad_embedding_forward(t->embed, idx);
    if (!h) return NULL;
    for (int32_t i = 0; i < t->n_blocks; i++) {
        TriadTensor *nh = triad_transformer_block_forward(t->blocks[i], h);
        if (!nh) return NULL;
        h = nh;
    }
    return triad_layer_norm_forward(t->ln_final, h);
}

int32_t triad_transformer_params(TriadTransformer *t, TriadTensor **out, int32_t max) {
    int32_t n = 0;
    if (n < max) out[n++] = t->embed->weight;
    for (int32_t i = 0; i < t->n_blocks; i++)
        n += triad_transformer_block_params(t->blocks[i], out + n, max - n);
    if (n < max) out[n++] = t->ln_final->gamma;
    if (n < max) out[n++] = t->ln_final->beta;
    return n;
}

typedef struct {
    TriadTensor *x, *wr, *wi;
    double *a, *b;
    int64_t rows;
    int32_t in_f, out_f;
} WaveCtx;

static void _wave_ctx_free(void *ptr) {
    WaveCtx *c = (WaveCtx*)ptr;
    if (!c) return;
    free(c->a); free(c->b); free(c);
}

static void _wave_backward(TriadTensor *out) {
    WaveCtx *c = (WaveCtx*)out->_ctx;
    if (!c || !out->grad) return;
    if (c->wr->requires_grad && !c->wr->grad)
        c->wr->grad = calloc(c->wr->size, sizeof(double));
    if (c->wi->requires_grad && !c->wi->grad)
        c->wi->grad = calloc(c->wi->size, sizeof(double));
    if (c->x->requires_grad && !c->x->grad)
        c->x->grad = calloc(c->x->size, sizeof(double));
    int64_t rows = c->rows;
    int32_t in_f = c->in_f, out_f = c->out_f;
    for (int64_t i = 0; i < rows; i++) {
        for (int32_t j = 0; j < out_f; j++) {
            int64_t oi = i * out_f + j;
            double y = out->data[oi];
            if (y < 1e-12) y = 1e-12;
            double ga = out->grad[oi] * c->a[oi] / y;
            double gb = out->grad[oi] * c->b[oi] / y;
            for (int32_t k = 0; k < in_f; k++) {
                int64_t xi = i * (int64_t)in_f + k;
                int64_t wi_ = (int64_t)k * out_f + j;
                if (c->wr->requires_grad)
                    c->wr->grad[wi_] += c->x->data[xi] * ga;
                if (c->wi->requires_grad)
                    c->wi->grad[wi_] += c->x->data[xi] * gb;
                if (c->x->requires_grad)
                    c->x->grad[xi] += ga * c->wr->data[wi_]
                                    + gb * c->wi->data[wi_];
            }
        }
    }
}

TriadWavetriad *triad_wave_triad_new(int32_t in_f, int32_t out_f) {
    TriadWavetriad *w = malloc(sizeof(TriadWavetriad));
    w->in_features = in_f;
    w->out_features = out_f;
    int32_t shape[] = {in_f, out_f};
    w->wr = triad_tensor_randn(2, shape, 1);
    w->wi = triad_tensor_randn(2, shape, 1);
    double scale = 1.0 / sqrt((double)in_f);
    for (int64_t i = 0; i < w->wr->size; i++) {
        w->wr->data[i] *= scale;
        w->wi->data[i] *= scale;
    }
    return w;
}

void triad_wave_triad_free(TriadWavetriad *w) {
    if (!w) return;
    triad_tensor_free(w->wr);
    triad_tensor_free(w->wi);
    free(w);
}

TriadTensor *triad_wave_triad_forward(TriadWavetriad *w, TriadTensor *x) {
    int32_t in_f = w->in_features, out_f = w->out_features;
    if (x->ndim < 1 || x->shape[x->ndim - 1] != in_f) return NULL;
    int64_t rows = x->size / in_f;
    int32_t oshape[32];
    for (int32_t i = 0; i < x->ndim - 1; i++) oshape[i] = x->shape[i];
    oshape[x->ndim - 1] = out_f;
    TriadTensor *result = triad_tensor_new(x->ndim, oshape, 0);
    double *a = malloc((size_t)(rows * out_f) * sizeof(double));
    double *b = malloc((size_t)(rows * out_f) * sizeof(double));
    for (int64_t i = 0; i < rows; i++) {
        for (int32_t j = 0; j < out_f; j++) {
            double sa = 0, sb = 0;
            for (int32_t k = 0; k < in_f; k++) {
                double xv = x->data[i * (int64_t)in_f + k];
                sa += xv * w->wr->data[(int64_t)k * out_f + j];
                sb += xv * w->wi->data[(int64_t)k * out_f + j];
            }
            int64_t oi = i * out_f + j;
            a[oi] = sa; b[oi] = sb;
            result->data[oi] = sqrt(sa * sa + sb * sb + 1e-12);
        }
    }
    int needs = x->requires_grad || w->wr->requires_grad || w->wi->requires_grad;
    if (needs) {
        result->requires_grad = 1;
        result->nchildren = 3;
        result->children = malloc(3 * sizeof(TriadTensor*));
        result->children[0] = x; triad_tensor_retain(x);
        result->children[1] = w->wr; triad_tensor_retain(w->wr);
        result->children[2] = w->wi; triad_tensor_retain(w->wi);
        WaveCtx *ctx = malloc(sizeof(WaveCtx));
        ctx->x = x; ctx->wr = w->wr; ctx->wi = w->wi;
        ctx->a = a; ctx->b = b;
        ctx->rows = rows; ctx->in_f = in_f; ctx->out_f = out_f;
        result->_ctx = ctx;
        result->ctx_free = _wave_ctx_free;
        result->grad_fn = _wave_backward;
    } else {
        free(a); free(b);
    }
    return result;
}

int32_t triad_wave_triad_params(TriadWavetriad *w, TriadTensor **out, int32_t max) {
    int32_t n = 0;
    if (n < max) out[n++] = w->wr;
    if (n < max) out[n++] = w->wi;
    return n;
}
