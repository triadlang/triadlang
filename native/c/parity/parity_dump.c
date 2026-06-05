/* parity_dump.c — print C op outputs for fixed inputs, for Python parity check.
   Format: one line per case: "KEY v0 v1 v2 ..."  */
#include "triad_ml.h"
#include "triad_train.h"
#include <stdio.h>

static void dump(const char *key, const double *v, int64_t n) {
    printf("%s", key);
    for (int64_t i = 0; i < n; i++) printf(" %.10g", v[i]);
    printf("\n");
}

int main(void) {
    int32_t s23[] = {2, 3};
    double A[] = {1, 5, 2, 4, 0, 6};

    /* softmax over axis 0 and axis 1 */
    {
        TriadTensor *x = triad_tensor_from_data(2, s23, A, 0);
        TriadTensor *s0 = triad_tensor_softmax_axis(x, 0);
        TriadTensor *s1 = triad_tensor_softmax_axis(x, 1);
        dump("softmax_axis0", s0->data, 6);
        dump("softmax_axis1", s1->data, 6);
        triad_tensor_free(x); triad_tensor_free(s0); triad_tensor_free(s1);
    }
    /* max over axis1, min over axis0 */
    {
        TriadTensor *x = triad_tensor_from_data(2, s23, A, 0);
        TriadTensor *mx = triad_tensor_max_axis(x, 1, 0);
        TriadTensor *mn = triad_tensor_min_axis(x, 0, 0);
        dump("max_axis1", mx->data, 2);
        dump("min_axis0", mn->data, 3);
        triad_tensor_free(x); triad_tensor_free(mx); triad_tensor_free(mn);
    }
    /* cat axis0: A (2,3) + B (1,3) */
    {
        TriadTensor *a = triad_tensor_from_data(2, s23, A, 0);
        TriadTensor *b = triad_tensor_from_data(2, (int32_t[]){1,3}, (double[]){7,8,9}, 0);
        TriadTensor *xs[] = {a, b};
        TriadTensor *c = triad_tensor_cat(xs, 2, 0);
        dump("cat_axis0", c->data, 9);
        triad_tensor_free(a); triad_tensor_free(b); triad_tensor_free(c);
    }
    /* stack axis1: u,v (3,) -> (3,2) */
    {
        TriadTensor *u = triad_tensor_from_data(1, (int32_t[]){3}, (double[]){1,2,3}, 0);
        TriadTensor *v = triad_tensor_from_data(1, (int32_t[]){3}, (double[]){4,5,6}, 0);
        TriadTensor *xs[] = {u, v};
        TriadTensor *st = triad_tensor_stack(xs, 2, 1);
        dump("stack_axis1", st->data, 6);
        triad_tensor_free(u); triad_tensor_free(v); triad_tensor_free(st);
    }
    /* losses */
    {
        double prob[] = {0.2,0.8,0.5,0.9}, logit[] = {-1,2,0,1.5}, tgt[] = {0,1,0,1};
        int32_t s4[] = {4};
        TriadTensor *p = triad_tensor_from_data(1, s4, prob, 0);
        TriadTensor *lg = triad_tensor_from_data(1, s4, logit, 0);
        TriadTensor *t = triad_tensor_from_data(1, s4, tgt, 0);
        TriadTensor *bce = triad_tensor_bce_loss(p, t);
        TriadTensor *bcl = triad_tensor_bce_with_logits(lg, t);
        dump("bce", bce->data, 1);
        dump("bce_logits", bcl->data, 1);
        triad_tensor_free(p); triad_tensor_free(lg); triad_tensor_free(t);
        triad_tensor_free(bce); triad_tensor_free(bcl);

        int32_t s2c[] = {2, 3};
        double L[] = {1,2,0.5,-1,0,3}, cls[] = {2,0};
        TriadTensor *lo = triad_tensor_from_data(2, s2c, L, 0);
        TriadTensor *sm = triad_tensor_softmax_axis(lo, 1);
        TriadTensor *lsm = triad_tensor_log(sm);
        TriadTensor *cl = triad_tensor_from_data(1, (int32_t[]){2}, cls, 0);
        TriadTensor *nll = triad_tensor_nll_loss(lsm, cl);
        dump("nll", nll->data, 1);
        triad_tensor_free(lo); triad_tensor_free(sm); triad_tensor_free(lsm);
        triad_tensor_free(cl); triad_tensor_free(nll);
    }
    /* metrics */
    {
        int32_t s32[] = {3, 2};
        TriadTensor *pred = triad_tensor_from_data(2, s32,
                              (double[]){0.1,0.9, 0.8,0.2, 0.3,0.7}, 0);
        TriadTensor *tg = triad_tensor_from_data(1, (int32_t[]){3}, (double[]){1,0,0}, 0);
        double acc = triad_metric_accuracy(pred, tg);
        dump("accuracy", &acc, 1);
        triad_tensor_free(pred); triad_tensor_free(tg);

        int32_t s4[] = {4};
        TriadTensor *p = triad_tensor_from_data(1, s4, (double[]){1,2,3,4}, 0);
        TriadTensor *t = triad_tensor_from_data(1, s4, (double[]){1,2,3,5}, 0);
        double mae = triad_metric_mae(p, t), rmse = triad_metric_rmse(p, t);
        dump("mae", &mae, 1);
        dump("rmse", &rmse, 1);
        triad_tensor_free(p); triad_tensor_free(t);
    }
    return 0;
}
