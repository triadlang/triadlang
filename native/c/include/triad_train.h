#ifndef TRIAD_TRAIN_H
#define TRIAD_TRAIN_H

#include "triad_ml.h"

typedef struct {
    TriadTensor *x;
    TriadTensor *y;
    int64_t n;
} TriadDataset;

TriadDataset *triad_dataset_new(TriadTensor *x, TriadTensor *y);
void          triad_dataset_free(TriadDataset *d);
int64_t       triad_dataset_len(const TriadDataset *d);

typedef struct {
    TriadDataset *ds;
    int32_t   batch_size;
    int       shuffle;
    uint64_t  rng;
    int64_t  *order;
    int64_t   cursor;
} TriadDataLoader;

TriadDataLoader *triad_dataloader_new(TriadDataset *ds, int32_t batch_size,
                                      int shuffle, uint64_t seed);
void             triad_dataloader_free(TriadDataLoader *dl);
int32_t          triad_dataloader_num_batches(const TriadDataLoader *dl);
void             triad_dataloader_reset(TriadDataLoader *dl);

int32_t          triad_dataloader_next(TriadDataLoader *dl,
                                       TriadTensor **xb, TriadTensor **yb);

double triad_metric_accuracy(const TriadTensor *pred, const TriadTensor *target);

double triad_metric_mae(const TriadTensor *pred, const TriadTensor *target);
double triad_metric_rmse(const TriadTensor *pred, const TriadTensor *target);
double triad_metric_r2(const TriadTensor *pred, const TriadTensor *target);
double triad_metric_perplexity(double loss_value);

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

double triad_trainer_train_epoch(TriadTrainer *tr, TriadDataLoader *dl);

double triad_trainer_evaluate(TriadTrainer *tr, TriadDataLoader *dl);

#endif
