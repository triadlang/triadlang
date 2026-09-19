#include "triad_train.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint64_t _xs(uint64_t *s) {
    uint64_t x = *s ? *s : 0x9E3779B97F4A7C15ull;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

TriadDataset *triad_dataset_new(TriadTensor *x, TriadTensor *y) {
    if (!x || x->ndim < 1) return NULL;
    if (y && y->shape[0] != x->shape[0]) return NULL;
    TriadDataset *d = malloc(sizeof(TriadDataset));
    if (!d) return NULL;
    d->x = x; d->y = y; d->n = x->shape[0];
    return d;
}

void triad_dataset_free(TriadDataset *d) {
    if (!d) return;
    triad_tensor_free(d->x);
    if (d->y) triad_tensor_free(d->y);
    free(d);
}

int64_t triad_dataset_len(const TriadDataset *d) { return d ? d->n : 0; }

TriadDataLoader *triad_dataloader_new(TriadDataset *ds, int32_t batch_size,
                                      int shuffle, uint64_t seed) {
    if (!ds || batch_size < 1 || ds->n < 1) return NULL;
    TriadDataLoader *dl = malloc(sizeof(TriadDataLoader));
    if (!dl) return NULL;
    dl->ds = ds;
    dl->batch_size = batch_size;
    dl->shuffle = shuffle;
    dl->rng = seed ? seed : 0x123456789ABCDEF;
    dl->order = malloc((size_t)ds->n * sizeof(int64_t));
    if (!dl->order) { free(dl); return NULL; }
    dl->cursor = 0;
    triad_dataloader_reset(dl);
    return dl;
}

void triad_dataloader_free(TriadDataLoader *dl) {
    if (!dl) return;
    free(dl->order);
    free(dl);
}

int32_t triad_dataloader_num_batches(const TriadDataLoader *dl) {
    if (!dl) return 0;
    int64_t n = dl->ds->n, b = dl->batch_size;
    return (int32_t)((n + b - 1) / b);
}

void triad_dataloader_reset(TriadDataLoader *dl) {
    int64_t n = dl->ds->n;
    for (int64_t i = 0; i < n; i++) dl->order[i] = i;
    if (dl->shuffle) {
        for (int64_t i = n - 1; i > 0; i--) {
            int64_t j = (int64_t)(_xs(&dl->rng) % (uint64_t)(i + 1));
            int64_t t = dl->order[i]; dl->order[i] = dl->order[j]; dl->order[j] = t;
        }
    }
    dl->cursor = 0;
}

static TriadTensor *_gather_rows(const TriadTensor *src, const int64_t *idx, int32_t b) {
    int32_t shape[32];
    shape[0] = b;
    for (int32_t i = 1; i < src->ndim; i++) shape[i] = src->shape[i];
    int32_t ndim = src->ndim;

    TriadTensor *out = triad_tensor_new(ndim, shape, 0);
    int64_t row = src->size / src->shape[0];
    for (int32_t i = 0; i < b; i++)
        memcpy(out->data + (int64_t)i * row,
               src->data + idx[i] * row,
               (size_t)row * sizeof(double));
    return out;
}

int32_t triad_dataloader_next(TriadDataLoader *dl, TriadTensor **xb, TriadTensor **yb) {
    int64_t n = dl->ds->n;
    if (dl->cursor >= n) return 0;
    int32_t b = (int32_t)((n - dl->cursor < dl->batch_size) ? (n - dl->cursor)
                                                            : dl->batch_size);
    const int64_t *idx = dl->order + dl->cursor;
    if (xb) *xb = _gather_rows(dl->ds->x, idx, b);
    if (yb) *yb = dl->ds->y ? _gather_rows(dl->ds->y, idx, b) : NULL;
    dl->cursor += b;
    return b;
}

double triad_metric_accuracy(const TriadTensor *pred, const TriadTensor *target) {
    if (!pred || !target || pred->ndim < 1) return 0.0;
    int32_t C = pred->shape[pred->ndim - 1];
    int64_t n = pred->size / C;
    if (target->size != n) return 0.0;
    int64_t correct = 0;
    for (int64_t i = 0; i < n; i++) {
        int32_t arg = 0; double mx = pred->data[i * C];
        for (int32_t j = 1; j < C; j++)
            if (pred->data[i * C + j] > mx) { mx = pred->data[i * C + j]; arg = j; }
        if (arg == (int32_t)target->data[i]) correct++;
    }
    return (double)correct / (double)n;
}

double triad_metric_mae(const TriadTensor *pred, const TriadTensor *target) {
    if (!pred || !target || pred->size != target->size) return 0.0;
    double s = 0;
    for (int64_t i = 0; i < pred->size; i++) s += fabs(pred->data[i] - target->data[i]);
    return s / (double)pred->size;
}

double triad_metric_rmse(const TriadTensor *pred, const TriadTensor *target) {
    if (!pred || !target || pred->size != target->size) return 0.0;
    double s = 0;
    for (int64_t i = 0; i < pred->size; i++) {
        double d = pred->data[i] - target->data[i];
        s += d * d;
    }
    return sqrt(s / (double)pred->size);
}

double triad_metric_r2(const TriadTensor *pred, const TriadTensor *target) {
    if (!pred || !target || pred->size != target->size || pred->size == 0) return 0.0;
    double mean = 0;
    for (int64_t i = 0; i < target->size; i++) mean += target->data[i];
    mean /= (double)target->size;
    double ss_res = 0, ss_tot = 0;
    for (int64_t i = 0; i < pred->size; i++) {
        double dr = target->data[i] - pred->data[i];
        double dt = target->data[i] - mean;
        ss_res += dr * dr; ss_tot += dt * dt;
    }
    if (ss_tot == 0.0) return 0.0;
    return 1.0 - ss_res / ss_tot;
}

double triad_metric_perplexity(double loss_value) { return exp(loss_value); }

double triad_trainer_train_epoch(TriadTrainer *tr, TriadDataLoader *dl) {
    triad_dataloader_reset(dl);
    double total = 0.0; int32_t nb = 0;
    TriadTensor *xb, *yb;
    int32_t b;
    while ((b = triad_dataloader_next(dl, &xb, &yb)) > 0) {
        if (tr->opt_zero) tr->opt_zero(tr->opt);
        TriadTensor *pred = tr->forward(tr->model, xb);
        TriadTensor *loss = tr->loss_fn(pred, yb);
        total += loss->data[0];
        nb++;
        triad_tensor_backward(loss, NULL);
        if (tr->opt_step) tr->opt_step(tr->opt);
        triad_tensor_free(loss);
        triad_tensor_free(pred);
        triad_tensor_free(xb);
        if (yb) triad_tensor_free(yb);
    }
    return nb ? total / nb : 0.0;
}

double triad_trainer_evaluate(TriadTrainer *tr, TriadDataLoader *dl) {
    triad_dataloader_reset(dl);
    int prev = triad_ml_get_grad();
    triad_ml_set_grad(0);
    double total = 0.0; int32_t nb = 0;
    TriadTensor *xb, *yb;
    int32_t b;
    while ((b = triad_dataloader_next(dl, &xb, &yb)) > 0) {
        TriadTensor *pred = tr->forward(tr->model, xb);
        TriadTensor *loss = tr->loss_fn(pred, yb);
        total += loss->data[0];
        nb++;
        triad_tensor_free(loss);
        triad_tensor_free(pred);
        triad_tensor_free(xb);
        if (yb) triad_tensor_free(yb);
    }
    triad_ml_set_grad(prev);
    return nb ? total / nb : 0.0;
}
