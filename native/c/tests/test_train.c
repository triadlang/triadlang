/* test_train.c — Dataset/DataLoader/Trainer/Metrics (item 8) */
#include "triad_train.h"
#include "triad_rt.h"
#include <stdio.h>
#include <math.h>

static int g_fail = 0;
static void check(const char *name, int ok) {
    printf("  %-32s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
}

/* adapters so the generic Trainer can drive the concrete types */
static TriadTensor *seq_forward(void *m, TriadTensor *x) {
    return triad_sequential_forward((TriadSequential*)m, x);
}
static void adam_step(void *o) { triad_adam_step((TriadAdam*)o); }
static void adam_zero(void *o) { triad_adam_zero_grad((TriadAdam*)o); }

int main(void) {
    triad_ml_seed(3);
    printf("=== dataset/loader/trainer/metrics (item 8) ===\n");

    /* ── Metrics on known inputs ── */
    {
        int32_t s[] = {3, 2};
        /* preds: argmax = 1,0,1 ; targets 1,0,0 -> 2/3 correct */
        TriadTensor *pred = triad_tensor_from_data(2, s, (double[]){0.1,0.9, 0.8,0.2, 0.3,0.7}, 0);
        TriadTensor *tg = triad_tensor_from_data(1, (int32_t[]){3}, (double[]){1,0,0}, 0);
        check("accuracy", fabs(triad_metric_accuracy(pred, tg) - 2.0/3.0) < 1e-9);
        triad_tensor_free(pred); triad_tensor_free(tg);

        int32_t r[] = {4};
        TriadTensor *p = triad_tensor_from_data(1, r, (double[]){1,2,3,4}, 0);
        TriadTensor *t = triad_tensor_from_data(1, r, (double[]){1,2,3,5}, 0);
        check("mae", fabs(triad_metric_mae(p,t) - 0.25) < 1e-9);
        check("rmse", fabs(triad_metric_rmse(p,t) - 0.5) < 1e-9);
        check("r2 (perfect=1)", fabs(triad_metric_r2(p,p) - 1.0) < 1e-9);
        check("perplexity(0)=1", fabs(triad_metric_perplexity(0.0) - 1.0) < 1e-9);
        triad_tensor_free(p); triad_tensor_free(t);
    }

    /* ── Build a separable 2-class dataset: class0 ~ (-1,-1), class1 ~ (1,1) ── */
    int64_t N = 40;
    int32_t xs[] = {(int32_t)N, 2};
    TriadTensor *X = triad_tensor_new(2, xs, 0);
    TriadTensor *Y = triad_tensor_new(1, (int32_t[]){(int32_t)N}, 0);
    uint64_t rng = 99;
    for (int64_t i = 0; i < N; i++) {
        int cls = (int)(i % 2);
        double cx = cls ? 1.0 : -1.0;
        /* cheap deterministic jitter */
        rng = rng * 6364136223846793005ull + 1;
        double j1 = ((double)((rng >> 33) % 1000) / 1000.0 - 0.5) * 0.4;
        rng = rng * 6364136223846793005ull + 1;
        double j2 = ((double)((rng >> 33) % 1000) / 1000.0 - 0.5) * 0.4;
        X->data[i*2+0] = cx + j1;
        X->data[i*2+1] = cx + j2;
        Y->data[i] = cls;
    }
    TriadDataset *ds = triad_dataset_new(X, Y);
    check("dataset len", triad_dataset_len(ds) == N);

    /* ── DataLoader batching ── */
    {
        TriadDataLoader *dl = triad_dataloader_new(ds, 16, 1, 7);
        check("num_batches", triad_dataloader_num_batches(dl) == 3);
        triad_dataloader_reset(dl);
        int64_t seen = 0; int32_t nb = 0; int shape_ok = 1;
        TriadTensor *xb, *yb; int32_t b;
        while ((b = triad_dataloader_next(dl, &xb, &yb)) > 0) {
            if (xb->ndim != 2 || xb->shape[1] != 2 || xb->shape[0] != b) shape_ok = 0;
            if (!yb || yb->shape[0] != b) shape_ok = 0;
            seen += b; nb++;
            triad_tensor_free(xb); triad_tensor_free(yb);
        }
        check("loader covers all samples", seen == N && nb == 3);
        check("batch shapes", shape_ok);
        triad_dataloader_free(dl);
    }

    /* ── Trainer reduces loss + raises accuracy ── */
    {
        TriadSequential *model = triad_sequential_new(3);
        triad_sequential_set(model, 0, TRIAD_LAYER_LINEAR, triad_linear_new(2, 16, 1));
        triad_sequential_set(model, 1, TRIAD_LAYER_RELU, NULL);
        triad_sequential_set(model, 2, TRIAD_LAYER_LINEAR, triad_linear_new(16, 2, 1));
        TriadTensor *params[16];
        int32_t np = triad_sequential_params(model, params, 16);
        TriadAdam *opt = triad_adam_new(params, np, 0.05, 0.9, 0.999, 1e-8);

        TriadTrainer tr = {
            .model = model, .forward = seq_forward,
            .loss_fn = triad_tensor_cross_entropy,
            .opt = opt, .opt_step = adam_step, .opt_zero = adam_zero,
        };
        TriadDataLoader *dl = triad_dataloader_new(ds, 8, 1, 42);

        double first = 0, last = 0;
        for (int e = 0; e < 30; e++) {
            double l = triad_trainer_train_epoch(&tr, dl);
            if (e == 0) first = l;
            last = l;
        }
        printf("    train loss: %.4f -> %.4f\n", first, last);
        check("trainer reduces loss", last < first * 0.5);

        /* final accuracy on the full set (eval, no grad) */
        triad_ml_set_grad(0);
        TriadTensor *pred = triad_sequential_forward(model, ds->x);
        double acc = triad_metric_accuracy(pred, ds->y);
        triad_ml_set_grad(1);
        printf("    final accuracy: %.3f\n", acc);
        check("trainer learns (acc>0.9)", acc > 0.9);
        triad_tensor_free(pred);

        double evl = triad_trainer_evaluate(&tr, dl);
        check("evaluate returns finite loss", isfinite(evl));

        triad_dataloader_free(dl);
        triad_adam_free(opt);
        /* sequential_free frees its linear layers */
        triad_sequential_free(model);
    }

    triad_dataset_free(ds);
    printf("%s\n", g_fail ? "train: FAIL" : "train: PASS");
    return g_fail;
}
