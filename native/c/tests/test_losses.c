/* test_losses.c — extra native losses (item 7): values, grads, equivalences */
#include "triad_ml.h"
#include "triad_rt.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
static void check(const char *name, int ok) {
    printf("  %-32s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
}
static int approx(double a, double b, double tol) { return fabs(a - b) < tol; }

/* numerical grad of loss(pred,target) wrt pred[i], target fixed */
static int grad_check(TriadTensor *(*loss)(TriadTensor*, TriadTensor*),
                      int n, const double *pd, const double *td) {
    int32_t s[] = {n};
    TriadTensor *p = triad_tensor_from_data(1, s, pd, 1);
    TriadTensor *t = triad_tensor_from_data(1, s, td, 0);
    TriadTensor *l = loss(p, t);
    triad_tensor_backward(l, NULL);
    int ok = 1;
    double eps = 1e-6;
    for (int i = 0; i < n && ok; i++) {
        double tmp[64]; memcpy(tmp, pd, n * sizeof(double));
        tmp[i] += eps;
        TriadTensor *pp = triad_tensor_from_data(1, s, tmp, 0);
        TriadTensor *lp = loss(pp, t);
        tmp[i] -= 2 * eps;
        TriadTensor *pm = triad_tensor_from_data(1, s, tmp, 0);
        TriadTensor *lm = loss(pm, t);
        double num = (lp->data[0] - lm->data[0]) / (2 * eps);
        if (!approx(num, p->grad[i], 1e-4)) ok = 0;
        triad_tensor_free(pp); triad_tensor_free(lp);
        triad_tensor_free(pm); triad_tensor_free(lm);
    }
    triad_tensor_free(p); triad_tensor_free(t); triad_tensor_free(l);
    return ok;
}

int main(void) {
    printf("=== extra losses (item 7) ===\n");

    double prob[] = {0.2, 0.8, 0.5, 0.9};
    double logit[] = {-1.0, 2.0, 0.0, 1.5};
    double tgt[]  = {0.0, 1.0, 0.0, 1.0};

    /* BCE forward value vs reference */
    {
        int32_t s[] = {4};
        TriadTensor *p = triad_tensor_from_data(1, s, prob, 0);
        TriadTensor *t = triad_tensor_from_data(1, s, tgt, 0);
        TriadTensor *l = triad_tensor_bce_loss(p, t);
        double ref = 0;
        for (int i = 0; i < 4; i++)
            ref -= tgt[i]*log(prob[i]) + (1-tgt[i])*log(1-prob[i]);
        ref /= 4;
        check("bce value", approx(l->data[0], ref, 1e-9));
        triad_tensor_free(p); triad_tensor_free(t); triad_tensor_free(l);
    }
    check("bce grad", grad_check(triad_tensor_bce_loss, 4, prob, tgt));

    /* BCE-with-logits value == BCE(sigmoid(logits)) */
    {
        int32_t s[] = {4};
        TriadTensor *x = triad_tensor_from_data(1, s, logit, 0);
        TriadTensor *t = triad_tensor_from_data(1, s, tgt, 0);
        TriadTensor *lwl = triad_tensor_bce_with_logits(x, t);
        double sig[4];
        for (int i = 0; i < 4; i++) sig[i] = 1.0/(1.0+exp(-logit[i]));
        TriadTensor *sp = triad_tensor_from_data(1, s, sig, 0);
        TriadTensor *lb = triad_tensor_bce_loss(sp, t);
        check("bce_logits == bce(sigmoid)", approx(lwl->data[0], lb->data[0], 1e-9));
        triad_tensor_free(x); triad_tensor_free(t);
        triad_tensor_free(lwl); triad_tensor_free(sp); triad_tensor_free(lb);
    }
    check("bce_logits grad", grad_check(triad_tensor_bce_with_logits, 4, logit, tgt));

    /* NLL(log_softmax(logits)) == cross_entropy(logits) */
    {
        int32_t s[] = {2, 3};
        double L[] = {1.0, 2.0, 0.5, -1.0, 0.0, 3.0};
        double cls[] = {2, 0};
        TriadTensor *lg = triad_tensor_from_data(2, s, L, 1);
        int32_t cs[] = {2};
        TriadTensor *cl = triad_tensor_from_data(1, cs, cls, 0);
        TriadTensor *ce = triad_tensor_cross_entropy(lg, cl);

        /* log_softmax = log(softmax) */
        TriadTensor *lg2 = triad_tensor_from_data(2, s, L, 1);
        TriadTensor *sm = triad_tensor_softmax_axis(lg2, 1);
        TriadTensor *lsm = triad_tensor_log(sm);
        TriadTensor *nll = triad_tensor_nll_loss(lsm, cl);
        check("nll(log_softmax)==cross_entropy", approx(nll->data[0], ce->data[0], 1e-9));

        /* grad path through nll should reach logits */
        triad_tensor_backward(nll, NULL);
        check("nll grad reaches logits", lg2->grad != NULL);

        triad_tensor_free(lg); triad_tensor_free(cl); triad_tensor_free(ce);
        triad_tensor_free(lg2); triad_tensor_free(sm);
        triad_tensor_free(lsm); triad_tensor_free(nll);
    }

    printf("%s\n", g_fail ? "losses: FAIL" : "losses: PASS");
    return g_fail;
}
