/* ═══════════════════════════════════════════════════════════════════
   TriadLang Native — Dataset / DataLoader / Trainer / Metrics (item 8)
   ═══════════════════════════════════════════════════════════════════
   ML-surface only. Metrics are diagnostic observables, never field
   controllers. The Trainer optimizes provided parameters via a provided
   optimizer; it does NOT impose targets on the Triad PDE evolution.
   ═══════════════════════════════════════════════════════════════════ */
#ifndef TRIAD_TRAIN_H
#define TRIAD_TRAIN_H

#include "triad_ml.h"

/* ── Dataset: feature tensor x (N, ...) + target tensor y (N, ...) ── */
typedef struct {
    TriadTensor *x;     /* owned */
    TriadTensor *y;     /* owned, may be NULL */
    int64_t n;          /* number of samples = x->shape[0] */
} TriadDataset;

TriadDataset *triad_dataset_new(TriadTensor *x, TriadTensor *y); /* takes ownership */
void          triad_dataset_free(TriadDataset *d);
int64_t       triad_dataset_len(const TriadDataset *d);

/* ── DataLoader: mini-batch iterator with optional shuffle ── */
typedef struct {
    TriadDataset *ds;       /* not owned */
    int32_t   batch_size;
    int       shuffle;
    uint64_t  rng;
    int64_t  *order;        /* permutation of [0, n) */
    int64_t   cursor;
} TriadDataLoader;

TriadDataLoader *triad_dataloader_new(TriadDataset *ds, int32_t batch_size,
                                      int shuffle, uint64_t seed);
void             triad_dataloader_free(TriadDataLoader *dl);
int32_t          triad_dataloader_num_batches(const TriadDataLoader *dl);
void             triad_dataloader_reset(TriadDataLoader *dl);   /* reshuffle + rewind */
/* Produces the next batch into *xb (and *yb if targets exist). Caller frees the
   returned tensors. Returns the batch size, or 0 when the epoch is exhausted. */
int32_t          triad_dataloader_next(TriadDataLoader *dl,
                                       TriadTensor **xb, TriadTensor **yb);

/* ── Metrics (diagnostic observables) ── */
/* classification: logits/probs (n, C) vs class-index targets (n,) */
double triad_metric_accuracy(const TriadTensor *pred, const TriadTensor *target);
/* regression element-wise */
double triad_metric_mae(const TriadTensor *pred, const TriadTensor *target);
double triad_metric_rmse(const TriadTensor *pred, const TriadTensor *target);
double triad_metric_r2(const TriadTensor *pred, const TriadTensor *target);
double triad_metric_perplexity(double loss_value);

/* ── Trainer: generic over model/optimizer via callbacks ── */
typedef TriadTensor *(*TriadForwardFn)(void *model, TriadTensor *x);
typedef TriadTensor *(*TriadLossFn)(TriadTensor *pred, TriadTensor *target);
typedef void (*TriadOptStepFn)(void *opt);
typedef void (*TriadOptZeroFn)(void *opt);

typedef struct {
    void            *model;
    TriadForwardFn   forward;
    TriadLossFn      loss_fn;
    void            *opt;
    TriadOptStepFn   opt_step;
    TriadOptZeroFn   opt_zero;
} TriadTrainer;

/* One pass with gradient updates; returns mean batch loss. */
double triad_trainer_train_epoch(TriadTrainer *tr, TriadDataLoader *dl);
/* One pass without gradients; returns mean batch loss (diagnostic only). */
double triad_trainer_evaluate(TriadTrainer *tr, TriadDataLoader *dl);

#endif /* TRIAD_TRAIN_H */
