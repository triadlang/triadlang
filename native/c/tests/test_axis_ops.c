/* test_axis_ops.c — forward + autograd checks for axis ops (item 5) */
#include "triad_ml.h"
#include "triad_rt.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
static void check(const char *name, int ok) {
    printf("  %-28s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
}
static int approx(double a, double b) { return fabs(a - b) < 1e-9; }

/* numerical-gradient check of a scalar-loss(sum(f(x))) against autograd */
static int grad_check(TriadTensor *(*op)(TriadTensor*, int32_t, int), int32_t axis, int kd,
                      int32_t ndim, const int32_t *shape, const double *data) {
    TriadTensor *x = triad_tensor_from_data(ndim, shape, data, 1);
    TriadTensor *y = op(x, axis, kd);
    TriadTensor *loss = triad_tensor_sum(y);
    triad_tensor_backward(loss, NULL);

    int ok = 1;
    double eps = 1e-6;
    for (int64_t i = 0; i < x->size && ok; i++) {
        double tmp[64]; memcpy(tmp, data, x->size * sizeof(double));
        tmp[i] += eps;
        TriadTensor *xp = triad_tensor_from_data(ndim, shape, tmp, 0);
        TriadTensor *yp = op(xp, axis, kd);
        double lp = 0; for (int64_t k = 0; k < yp->size; k++) lp += yp->data[k];
        tmp[i] -= 2 * eps;
        TriadTensor *xm = triad_tensor_from_data(ndim, shape, tmp, 0);
        TriadTensor *ym = op(xm, axis, kd);
        double lm = 0; for (int64_t k = 0; k < ym->size; k++) lm += ym->data[k];
        double num = (lp - lm) / (2 * eps);
        if (fabs(num - x->grad[i]) > 1e-4) ok = 0;
        triad_tensor_free(xp); triad_tensor_free(yp);
        triad_tensor_free(xm); triad_tensor_free(ym);
    }
    triad_tensor_free(x); triad_tensor_free(y); triad_tensor_free(loss);
    return ok;
}

int main(void) {
    triad_ml_seed(1);
    printf("=== axis ops (item 5) ===\n");

    int32_t s23[] = {2, 3};
    double d23[] = {1, 5, 2, 4, 0, 6};

    /* max_axis forward: axis=1 -> [5, 6] */
    {
        TriadTensor *x = triad_tensor_from_data(2, s23, d23, 0);
        TriadTensor *m = triad_tensor_max_axis(x, 1, 0);
        check("max_axis fwd", m->ndim == 1 && m->shape[0] == 2 &&
                              approx(m->data[0], 5) && approx(m->data[1], 6));
        triad_tensor_free(x); triad_tensor_free(m);
    }
    /* min_axis forward: axis=0 -> [1, 0, 2] */
    {
        TriadTensor *x = triad_tensor_from_data(2, s23, d23, 0);
        TriadTensor *m = triad_tensor_min_axis(x, 0, 0);
        check("min_axis fwd", m->ndim == 1 && m->shape[0] == 3 &&
                              approx(m->data[0], 1) && approx(m->data[1], 0) &&
                              approx(m->data[2], 2));
        triad_tensor_free(x); triad_tensor_free(m);
    }
    check("max_axis grad", grad_check(triad_tensor_max_axis, 1, 0, 2, s23, d23));
    check("min_axis grad", grad_check(triad_tensor_min_axis, 0, 1, 2, s23, d23));

    /* softmax_axis: axis=0 must sum to 1 per column; compare last-axis vs builtin */
    {
        TriadTensor *x = triad_tensor_from_data(2, s23, d23, 0);
        TriadTensor *sm = triad_tensor_softmax_axis(x, 0);
        double c0 = sm->data[0] + sm->data[3];
        check("softmax_axis col-sum", approx(c0, 1.0));
        TriadTensor *x2 = triad_tensor_from_data(2, s23, d23, 0);
        TriadTensor *a = triad_tensor_softmax_axis(x2, 1);
        TriadTensor *b = triad_tensor_softmax(x2); /* last axis */
        int eq = 1;
        for (int64_t i = 0; i < a->size; i++) if (!approx(a->data[i], b->data[i])) eq = 0;
        check("softmax_axis==softmax", eq);
        triad_tensor_free(x); triad_tensor_free(sm);
        triad_tensor_free(x2); triad_tensor_free(a); triad_tensor_free(b);
    }
    /* softmax_axis grad (loss=sum gives ~0 grad; use weighted loss) */
    {
        TriadTensor *x = triad_tensor_from_data(2, s23, d23, 1);
        TriadTensor *sm = triad_tensor_softmax_axis(x, 1);
        TriadTensor *w = triad_tensor_from_data(2, s23, (double[]){1,2,3,4,5,6}, 0);
        TriadTensor *prod = triad_tensor_mul(sm, w);
        TriadTensor *loss = triad_tensor_sum(prod);
        triad_tensor_backward(loss, NULL);
        /* numerical check element 0 */
        double eps = 1e-6, base[6]; memcpy(base, d23, sizeof base);
        double wv[] = {1,2,3,4,5,6};
        double ls[2];
        for (int sgn = 0; sgn < 2; sgn++) {
            double t[6]; memcpy(t, base, sizeof t);
            t[0] += sgn ? -eps : eps;
            TriadTensor *xx = triad_tensor_from_data(2, s23, t, 0);
            TriadTensor *ss = triad_tensor_softmax_axis(xx, 1);
            double l = 0; for (int k = 0; k < 6; k++) l += ss->data[k] * wv[k];
            ls[sgn] = l;
            triad_tensor_free(xx); triad_tensor_free(ss);
        }
        double num = (ls[0] - ls[1]) / (2 * eps);
        check("softmax_axis grad", fabs(num - x->grad[0]) < 1e-4);
        triad_tensor_free(x); triad_tensor_free(sm); triad_tensor_free(w);
        triad_tensor_free(prod); triad_tensor_free(loss);
    }

    /* cat along axis 0: (2,3)+(1,3) -> (3,3) */
    {
        int32_t s13[] = {1, 3};
        TriadTensor *a = triad_tensor_from_data(2, s23, d23, 1);
        TriadTensor *b = triad_tensor_from_data(2, s13, (double[]){7,8,9}, 1);
        TriadTensor *xs[] = {a, b};
        TriadTensor *c = triad_tensor_cat(xs, 2, 0);
        check("cat fwd shape", c->ndim == 2 && c->shape[0] == 3 && c->shape[1] == 3 &&
                               approx(c->data[6], 7) && approx(c->data[8], 9));
        TriadTensor *loss = triad_tensor_sum(c);
        triad_tensor_backward(loss, NULL);
        check("cat grad", a->grad && approx(a->grad[0], 1) && b->grad && approx(b->grad[2], 1));
        triad_tensor_free(a); triad_tensor_free(b);
        triad_tensor_free(c); triad_tensor_free(loss);
    }
    /* cat along axis 1: (2,3)+(2,2) -> (2,5) */
    {
        int32_t s22[] = {2, 2};
        TriadTensor *a = triad_tensor_from_data(2, s23, d23, 0);
        TriadTensor *b = triad_tensor_from_data(2, s22, (double[]){10,11,12,13}, 0);
        TriadTensor *xs[] = {a, b};
        TriadTensor *c = triad_tensor_cat(xs, 2, 1);
        /* row0: 1 5 2 10 11 ; row1: 4 0 6 12 13 */
        check("cat axis1 fwd", c->shape[1] == 5 &&
                               approx(c->data[3], 10) && approx(c->data[4], 11) &&
                               approx(c->data[8], 12) && approx(c->data[9], 13));
        triad_tensor_free(a); triad_tensor_free(b); triad_tensor_free(c);
    }
    /* stack along axis 0: two (3,) -> (2,3) */
    {
        int32_t s3[] = {3};
        TriadTensor *a = triad_tensor_from_data(1, s3, (double[]){1,2,3}, 1);
        TriadTensor *b = triad_tensor_from_data(1, s3, (double[]){4,5,6}, 1);
        TriadTensor *xs[] = {a, b};
        TriadTensor *c = triad_tensor_stack(xs, 2, 0);
        check("stack fwd", c->ndim == 2 && c->shape[0] == 2 && c->shape[1] == 3 &&
                           approx(c->data[0], 1) && approx(c->data[5], 6));
        TriadTensor *w = triad_tensor_from_data(2, (int32_t[]){2,3},
                                                (double[]){1,2,3,4,5,6}, 0);
        TriadTensor *prod = triad_tensor_mul(c, w);
        TriadTensor *loss = triad_tensor_sum(prod);
        triad_tensor_backward(loss, NULL);
        /* grad of a = w row0 = [1,2,3], grad of b = w row1 = [4,5,6] */
        check("stack grad", approx(a->grad[0],1) && approx(a->grad[2],3) &&
                            approx(b->grad[0],4) && approx(b->grad[2],6));
        triad_tensor_free(a); triad_tensor_free(b); triad_tensor_free(c);
        triad_tensor_free(w); triad_tensor_free(prod); triad_tensor_free(loss);
    }
    /* stack along axis 1: two (3,) -> (3,2) */
    {
        int32_t s3[] = {3};
        TriadTensor *a = triad_tensor_from_data(1, s3, (double[]){1,2,3}, 0);
        TriadTensor *b = triad_tensor_from_data(1, s3, (double[]){4,5,6}, 0);
        TriadTensor *xs[] = {a, b};
        TriadTensor *c = triad_tensor_stack(xs, 2, 1);
        /* expect [[1,4],[2,5],[3,6]] */
        check("stack axis1 fwd", c->shape[0] == 3 && c->shape[1] == 2 &&
                                 approx(c->data[0],1) && approx(c->data[1],4) &&
                                 approx(c->data[2],2) && approx(c->data[5],6));
        triad_tensor_free(a); triad_tensor_free(b); triad_tensor_free(c);
    }

    printf("%s\n", g_fail ? "axis ops: FAIL" : "axis ops: PASS");
    return g_fail;
}
