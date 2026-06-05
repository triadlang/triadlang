#include "triad_ml.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

static void check(const char *label, int ok) {
    printf("  %-28s %s\n", label, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
}

int main(void) {
    printf("=== dropout/batchnorm ===\n");

    {
        int32_t shape[] = {4};
        double data[] = {1, 2, 3, 4};
        TriadTensor *x = triad_tensor_from_data(1, shape, data, 1);
        TriadTensor *y = triad_tensor_dropout(x, 0.0, 1);
        check("dropout p0 forward", y && y->data[0] == 1 && y->data[3] == 4);
        TriadTensor *loss = triad_tensor_sum(y);
        triad_tensor_backward(loss, NULL);
        check("dropout p0 grad", x->grad && x->grad[0] == 1 && x->grad[3] == 1);
        triad_tensor_free(loss);
        triad_tensor_free(y);
        triad_tensor_free(x);
    }

    {
        int32_t shape[] = {4};
        double data[] = {1, 1, 1, 1};
        triad_ml_seed(123);
        TriadTensor *x = triad_tensor_from_data(1, shape, data, 0);
        TriadTensor *y = triad_tensor_dropout(x, 0.5, 1);
        int valid = y != NULL;
        for (int64_t i = 0; y && i < y->size; i++)
            if (!(y->data[i] == 0.0 || fabs(y->data[i] - 2.0) < 1e-12)) valid = 0;
        check("dropout mask scale", valid);
        triad_tensor_free(y);
        triad_tensor_free(x);
    }

    {
        int32_t shape[] = {4, 3};
        double data[] = {
            1, 2, 3,
            2, 3, 4,
            3, 4, 5,
            4, 5, 6
        };
        TriadTensor *x = triad_tensor_from_data(2, shape, data, 1);
        TriadBatchNorm1d *bn = triad_batch_norm1d_new(3, 1e-5, 0.1);
        TriadTensor *y = triad_batch_norm1d_forward(bn, x);
        int shape_ok = y && y->ndim == 2 && y->shape[0] == 4 && y->shape[1] == 3;
        check("batchnorm shape", shape_ok);
        int mean_ok = 1;
        for (int32_t j = 0; y && j < 3; j++) {
            double m = 0.0;
            for (int32_t i = 0; i < 4; i++) m += y->data[i * 3 + j];
            if (fabs(m / 4.0) > 1e-9) mean_ok = 0;
        }
        check("batchnorm mean", mean_ok);
        TriadTensor *loss = triad_tensor_sum(y);
        triad_tensor_backward(loss, NULL);
        check("batchnorm grad", x->grad != NULL && bn->gamma->grad != NULL && bn->beta->grad != NULL);
        triad_tensor_free(loss);
        triad_tensor_free(y);
        triad_batch_norm1d_free(bn);
        triad_tensor_free(x);
    }

    printf("dropout/batchnorm: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
