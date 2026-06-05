#include "triad_ml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

static void check(const char *label, int ok) {
    printf("  %-28s %s\n", label, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
}

int main(void) {
    printf("=== index/slice/compare ops ===\n");

    {
        int32_t shape[] = {2, 5};
        double data[] = {
            0, 1, 2, 3, 4,
            5, 6, 7, 8, 9
        };
        TriadTensor *x = triad_tensor_from_data(2, shape, data, 1);
        TriadTensor *s = triad_tensor_slice_axis(x, 1, 1, 5, 2);
        check("slice shape", s && s->ndim == 2 && s->shape[0] == 2 && s->shape[1] == 2);
        check("slice forward", s && s->data[0] == 1 && s->data[1] == 3 &&
                               s->data[2] == 6 && s->data[3] == 8);
        TriadTensor *loss = triad_tensor_sum(s);
        triad_tensor_backward(loss, NULL);
        check("slice grad scatter", x->grad &&
              x->grad[0] == 0 && x->grad[1] == 1 && x->grad[2] == 0 &&
              x->grad[3] == 1 && x->grad[4] == 0 &&
              x->grad[5] == 0 && x->grad[6] == 1 && x->grad[7] == 0 &&
              x->grad[8] == 1 && x->grad[9] == 0);
        triad_tensor_free(loss);
        triad_tensor_free(s);
        triad_tensor_free(x);
    }

    {
        int32_t shape[] = {4};
        double a_data[] = {1, 2, 3, 4};
        double b_data[] = {2, 2, 2, 2};
        TriadTensor *a = triad_tensor_from_data(1, shape, a_data, 1);
        TriadTensor *b = triad_tensor_from_data(1, shape, b_data, 1);
        TriadTensor *lt = triad_tensor_lt(a, b);
        TriadTensor *ge = triad_tensor_ge(a, b);
        check("lt mask", lt && lt->requires_grad == 0 &&
              lt->data[0] == 1 && lt->data[1] == 0 && lt->data[2] == 0 && lt->data[3] == 0);
        check("ge mask", ge && ge->requires_grad == 0 &&
              ge->data[0] == 0 && ge->data[1] == 1 && ge->data[2] == 1 && ge->data[3] == 1);
        triad_tensor_free(lt);
        triad_tensor_free(ge);
        triad_tensor_free(a);
        triad_tensor_free(b);
    }

    {
        int32_t shape[] = {3};
        double a_data[] = {1, 2, 3};
        TriadTensor *a = triad_tensor_from_data(1, shape, a_data, 0);
        TriadTensor *scalar = triad_tensor_scalar(2, 0);
        TriadTensor *gt = triad_tensor_gt(a, scalar);
        check("scalar compare", gt && gt->shape[0] == 3 &&
              gt->data[0] == 0 && gt->data[1] == 0 && gt->data[2] == 1);
        triad_tensor_free(gt);
        triad_tensor_free(scalar);
        triad_tensor_free(a);
    }

    printf("index ops: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
