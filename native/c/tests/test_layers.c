/* test_layers.c — composite NN layers (item 6): shapes, grad flow, learning */
#include "triad_ml.h"
#include "triad_rt.h"
#include <stdio.h>
#include <math.h>

static int g_fail = 0;
static void check(const char *name, int ok) {
    printf("  %-30s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
}

static int all_have_grad(TriadTensor **p, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        if (!p[i]->grad) return 0;
        int nz = 0;
        for (int64_t k = 0; k < p[i]->size; k++) if (fabs(p[i]->grad[k]) > 0) { nz = 1; break; }
        if (!nz) return 0;
    }
    return 1;
}

int main(void) {
    triad_ml_seed(7);
    printf("=== composite layers (item 6) ===\n");

    /* FeedForward */
    {
        TriadFeedForward *ff = triad_feedforward_new(4, 8);
        int32_t s[] = {2, 4};
        TriadTensor *x = triad_tensor_randn(2, s, 1);
        TriadTensor *y = triad_feedforward_forward(ff, x);
        check("ff fwd shape", y && y->ndim == 2 && y->shape[0] == 2 && y->shape[1] == 4);
        TriadTensor *loss = triad_tensor_sum(y);
        triad_tensor_backward(loss, NULL);
        TriadTensor *p[8]; int32_t n = triad_feedforward_params(ff, p, 8);
        check("ff grad flow", all_have_grad(p, n));
        triad_tensor_free(x); triad_tensor_free(y); triad_tensor_free(loss);
        triad_feedforward_free(ff);
    }

    /* MultiHeadAttention — 3D and 2D */
    {
        TriadMultiHeadAttention *m = triad_mha_new(8, 2);
        int32_t s[] = {1, 3, 8};
        TriadTensor *x = triad_tensor_randn(3, s, 1);
        TriadTensor *y = triad_mha_forward(m, x);
        check("mha 3D fwd shape", y && y->ndim == 3 && y->shape[0] == 1 &&
                                  y->shape[1] == 3 && y->shape[2] == 8);
        TriadTensor *loss = triad_tensor_sum(y);
        triad_tensor_backward(loss, NULL);
        TriadTensor *p[12]; int32_t n = triad_mha_params(m, p, 12);
        check("mha grad flow", all_have_grad(p, n));
        triad_tensor_free(x); triad_tensor_free(y); triad_tensor_free(loss);

        int32_t s2[] = {3, 8};
        TriadTensor *x2 = triad_tensor_randn(2, s2, 0);
        TriadTensor *y2 = triad_mha_forward(m, x2);
        check("mha 2D fwd shape", y2 && y2->ndim == 2 && y2->shape[0] == 3 && y2->shape[1] == 8);
        triad_tensor_free(x2); triad_tensor_free(y2);
        triad_mha_free(m);
    }

    /* TransformerBlock — shape preserved, grad to all params */
    {
        TriadTransformerBlock *b = triad_transformer_block_new(8, 2, 16);
        int32_t s[] = {1, 4, 8};
        TriadTensor *x = triad_tensor_randn(3, s, 1);
        TriadTensor *y = triad_transformer_block_forward(b, x);
        check("block fwd shape", y && y->ndim == 3 && y->shape[1] == 4 && y->shape[2] == 8);
        TriadTensor *loss = triad_tensor_sum(y);
        triad_tensor_backward(loss, NULL);
        TriadTensor *p[64]; int32_t n = triad_transformer_block_params(b, p, 64);
        check("block grad flow", n > 0 && all_have_grad(p, n));
        triad_tensor_free(x); triad_tensor_free(y); triad_tensor_free(loss);
        triad_transformer_block_free(b);
    }

    /* Transformer — token ids -> hidden states */
    {
        TriadTransformer *t = triad_transformer_new(10, 8, 2, 2, 16);
        int32_t s[] = {1, 4};
        TriadTensor *idx = triad_tensor_from_data(2, s, (double[]){1,2,3,4}, 0);
        TriadTensor *h = triad_transformer_forward(t, idx);
        check("transformer fwd shape", h && h->ndim == 3 && h->shape[0] == 1 &&
                                       h->shape[1] == 4 && h->shape[2] == 8);
        TriadTensor *loss = triad_tensor_sum(h);
        triad_tensor_backward(loss, NULL);
        check("transformer embed grad", t->embed->weight->grad != NULL);
        triad_tensor_free(idx); triad_tensor_free(h); triad_tensor_free(loss);
        triad_transformer_free(t);
    }

    /* Learning: a transformer block must reduce MSE toward a fixed target */
    {
        TriadTransformerBlock *b = triad_transformer_block_new(8, 2, 16);
        TriadTensor *p[64]; int32_t n = triad_transformer_block_params(b, p, 64);
        TriadAdam *opt = triad_adam_new(p, n, 0.01, 0.9, 0.999, 1e-8);
        int32_t s[] = {1, 4, 8};
        double first = 0, last = 0;
        TriadTensor *x = triad_tensor_randn(3, s, 0);
        TriadTensor *target = triad_tensor_zeros(3, s, 0);
        for (int step = 0; step < 60; step++) {
            triad_adam_zero_grad(opt);
            TriadTensor *y = triad_transformer_block_forward(b, x);
            TriadTensor *loss = triad_tensor_mse_loss(y, target);
            double lv = loss->data[0];
            if (step == 0) first = lv;
            last = lv;
            triad_tensor_backward(loss, NULL);
            triad_adam_step(opt);
            triad_tensor_free(y); triad_tensor_free(loss);
        }
        printf("    loss: %.5f -> %.5f\n", first, last);
        check("block learns (loss down)", last < first * 0.5);
        triad_tensor_free(x); triad_tensor_free(target);
        triad_adam_free(opt); triad_transformer_block_free(b);
    }

    printf("%s\n", g_fail ? "layers: FAIL" : "layers: PASS");
    return g_fail;
}
