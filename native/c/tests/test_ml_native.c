/* test_ml_native.c — XOR training test for native ML */
#include "triad_ml.h"
#include "triad_rt.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    triad_ml_seed(42);

    /* ── Test 1: tensor basics ── */
    {
        int32_t s[] = {2, 3};
        TriadTensor *a = triad_tensor_ones(2, s, 0);
        TriadTensor *b = triad_tensor_from_data(2, s, (double[]){1,2,3,4,5,6}, 0);
        TriadTensor *c = triad_tensor_add(a, b);
        /* c should be {2,3,4,5,6,7} */
        int ok = (c->data[0] == 2.0 && c->data[5] == 7.0);
        printf("tensor add:    %s\n", ok ? "PASS" : "FAIL");
        triad_tensor_free(a); triad_tensor_free(b); triad_tensor_free(c);
    }

    /* ── Test 2: autograd add ── */
    {
        int32_t s[] = {3};
        TriadTensor *a = triad_tensor_from_data(1, s, (double[]){1,2,3}, 1);
        TriadTensor *b = triad_tensor_from_data(1, s, (double[]){4,5,6}, 1);
        TriadTensor *c = triad_tensor_add(a, b);
        TriadTensor *loss = triad_tensor_sum(c);
        triad_tensor_backward(loss, NULL);
        /* grad_a = [1,1,1], grad_b = [1,1,1] */
        int ok = (a->grad[0] == 1.0 && a->grad[2] == 1.0 &&
                  b->grad[0] == 1.0 && b->grad[2] == 1.0);
        printf("autograd add:  %s\n", ok ? "PASS" : "FAIL");
        triad_tensor_free(a); triad_tensor_free(b);
        triad_tensor_free(c); triad_tensor_free(loss);
    }

    /* ── Test 3: autograd mul ── */
    {
        int32_t s[] = {2};
        TriadTensor *a = triad_tensor_from_data(1, s, (double[]){3, 4}, 1);
        TriadTensor *b = triad_tensor_from_data(1, s, (double[]){2, 5}, 1);
        TriadTensor *c = triad_tensor_mul(a, b);
        TriadTensor *loss = triad_tensor_sum(c);
        triad_tensor_backward(loss, NULL);
        /* grad_a = b = [2,5], grad_b = a = [3,4] */
        int ok = (fabs(a->grad[0] - 2.0) < 1e-10 && fabs(a->grad[1] - 5.0) < 1e-10 &&
                  fabs(b->grad[0] - 3.0) < 1e-10 && fabs(b->grad[1] - 4.0) < 1e-10);
        printf("autograd mul:  %s\n", ok ? "PASS" : "FAIL");
        triad_tensor_free(a); triad_tensor_free(b);
        triad_tensor_free(c); triad_tensor_free(loss);
    }

    /* ── Test 4: matmul + backward ── */
    {
        int32_t sa[] = {2, 3}, sb[] = {3, 2};
        TriadTensor *a = triad_tensor_from_data(2, sa, (double[]){1,2,3,4,5,6}, 1);
        TriadTensor *b = triad_tensor_from_data(2, sb, (double[]){1,0,0,1,1,0}, 1);
        TriadTensor *c = triad_tensor_matmul(a, b);
        /* c = [[4,2],[10,5]] */
        int ok = (fabs(c->data[0] - 4.0) < 1e-10 && fabs(c->data[3] - 5.0) < 1e-10);
        TriadTensor *loss = triad_tensor_sum(c);
        triad_tensor_backward(loss, NULL);
        printf("matmul fwd:    %s\n", ok ? "PASS" : "FAIL");
        printf("matmul grad:   %s\n", (a->grad != NULL && b->grad != NULL) ? "PASS" : "FAIL");
        triad_tensor_free(a); triad_tensor_free(b);
        triad_tensor_free(c); triad_tensor_free(loss);
    }

    /* ── Test 5: sigmoid + relu ── */
    {
        int32_t s[] = {3};
        TriadTensor *a = triad_tensor_from_data(1, s, (double[]){-1, 0, 1}, 1);
        TriadTensor *sig = triad_tensor_sigmoid(a);
        /* sigmoid(-1) ≈ 0.2689, sigmoid(0) = 0.5, sigmoid(1) ≈ 0.7311 */
        int ok = (fabs(sig->data[1] - 0.5) < 1e-10 && sig->data[2] > 0.7 && sig->data[0] < 0.3);
        printf("sigmoid:       %s\n", ok ? "PASS" : "FAIL");

        TriadTensor *b = triad_tensor_from_data(1, s, (double[]){-2, 0, 3}, 0);
        TriadTensor *r = triad_tensor_relu(b);
        ok = (r->data[0] == 0.0 && r->data[1] == 0.0 && r->data[2] == 3.0);
        printf("relu:          %s\n", ok ? "PASS" : "FAIL");
        triad_tensor_free(a); triad_tensor_free(sig);
        triad_tensor_free(b); triad_tensor_free(r);
    }

    /* ── Test 6: shape ops + batched matmul ── */
    {
        int32_t s[] = {2, 3};
        TriadTensor *a = triad_tensor_from_data(2, s, (double[]){1,2,3,4,5,6}, 1);
        int32_t rs[] = {3, 2};
        TriadTensor *r = triad_tensor_reshape(a, 2, rs);
        TriadTensor *t = triad_tensor_transpose(r, 0, 1);
        TriadTensor *loss = triad_tensor_sum(t);
        triad_tensor_backward(loss, NULL);
        int ok = (r && t && t->shape[0] == 2 && t->shape[1] == 3 &&
                  fabs(t->data[0] - 1.0) < 1e-10 &&
                  fabs(t->data[1] - 3.0) < 1e-10 &&
                  a->grad && fabs(a->grad[5] - 1.0) < 1e-10);
        printf("shape ops:     %s\n", ok ? "PASS" : "FAIL");
        triad_tensor_free(a); triad_tensor_free(r);
        triad_tensor_free(t); triad_tensor_free(loss);

        int32_t ba[] = {2, 2, 3}, bb[] = {2, 3, 2};
        TriadTensor *x = triad_tensor_from_data(3, ba,
            (double[]){1,2,3,4,5,6, 1,0,0,1,1,0}, 1);
        TriadTensor *y = triad_tensor_from_data(3, bb,
            (double[]){1,0,0,1,1,0, 2,1,1,0,0,1}, 1);
        TriadTensor *z = triad_tensor_bmm(x, y);
        TriadTensor *zloss = triad_tensor_sum(z);
        triad_tensor_backward(zloss, NULL);
        ok = (z && z->ndim == 3 && z->shape[0] == 2 && z->shape[1] == 2 &&
              z->shape[2] == 2 && fabs(z->data[0] - 4.0) < 1e-10 &&
              x->grad && y->grad);
        printf("bmm:           %s\n", ok ? "PASS" : "FAIL");
        triad_tensor_free(x); triad_tensor_free(y);
        triad_tensor_free(z); triad_tensor_free(zloss);
    }

    /* ── Test 7: sqrt + layer norm ── */
    {
        int32_t s[] = {2, 3};
        TriadTensor *a = triad_tensor_from_data(2, s, (double[]){1,4,9,2,3,4}, 1);
        TriadTensor *sq = triad_tensor_sqrt(a);
        int32_t gs[] = {3};
        TriadTensor *g = triad_tensor_ones(1, gs, 1);
        TriadTensor *b = triad_tensor_zeros(1, gs, 1);
        TriadTensor *ln = triad_tensor_layer_norm(a, g, b, 1e-5);
        TriadTensor *loss = triad_tensor_sum(ln);
        triad_tensor_backward(loss, NULL);
        int ok = (sq && fabs(sq->data[2] - 3.0) < 1e-10 &&
                  ln && ln->shape[0] == 2 && ln->shape[1] == 3 &&
                  a->grad && g->grad && b->grad);
        printf("layer_norm:    %s\n", ok ? "PASS" : "FAIL");
        triad_tensor_free(a); triad_tensor_free(sq);
        triad_tensor_free(g); triad_tensor_free(b);
        triad_tensor_free(ln); triad_tensor_free(loss);
    }

    /* ── Test 8: axis reductions ── */
    {
        int32_t s[] = {2, 3};
        TriadTensor *a = triad_tensor_from_data(2, s, (double[]){1,2,3,4,5,6}, 1);
        TriadTensor *sum0 = triad_tensor_sum_axis(a, 0, 0);
        TriadTensor *mean1 = triad_tensor_mean_axis(a, 1, 1);
        TriadTensor *loss = triad_tensor_sum(mean1);
        triad_tensor_backward(loss, NULL);
        int ok = (sum0 && sum0->ndim == 1 && sum0->shape[0] == 3 &&
                  fabs(sum0->data[0] - 5.0) < 1e-10 &&
                  fabs(sum0->data[2] - 9.0) < 1e-10 &&
                  mean1 && mean1->ndim == 2 && mean1->shape[0] == 2 &&
                  mean1->shape[1] == 1 &&
                  fabs(mean1->data[0] - 2.0) < 1e-10 &&
                  fabs(mean1->data[1] - 5.0) < 1e-10 &&
                  a->grad && fabs(a->grad[0] - 1.0/3.0) < 1e-10);
        printf("axis reduce:   %s\n", ok ? "PASS" : "FAIL");
        triad_tensor_free(a); triad_tensor_free(sum0);
        triad_tensor_free(mean1); triad_tensor_free(loss);
    }

    /* ── Test 9: cross-entropy ── */
    {
        int32_t sl[] = {2, 3}, st[] = {2};
        TriadTensor *logits = triad_tensor_from_data(2, sl, (double[]){2,1,0, 0,1,2}, 1);
        TriadTensor *targets = triad_tensor_from_data(1, st, (double[]){0, 2}, 0);
        TriadTensor *loss = triad_tensor_cross_entropy(logits, targets);
        /* loss should be reasonable (< 2.0) */
        int ok = (loss->data[0] > 0 && loss->data[0] < 2.0);
        triad_tensor_backward(loss, NULL);
        ok = ok && (logits->grad != NULL);
        printf("cross_entropy: %s (loss=%.4f)\n", ok ? "PASS" : "FAIL", loss->data[0]);
        triad_tensor_free(logits); triad_tensor_free(targets); triad_tensor_free(loss);
    }

    /* ── Test 10: regression losses ── */
    {
        int32_t s[] = {3};
        TriadTensor *p = triad_tensor_from_data(1, s, (double[]){1, 3, 6}, 1);
        TriadTensor *t = triad_tensor_from_data(1, s, (double[]){2, 1, 4}, 1);
        TriadTensor *l1 = triad_tensor_l1_loss(p, t);
        TriadTensor *hub = triad_tensor_huber_loss(p, t, 1.0);
        triad_tensor_backward(l1, NULL);
        int ok = (l1 && fabs(l1->data[0] - (5.0/3.0)) < 1e-10 &&
                  hub && hub->data[0] > 0.0 &&
                  p->grad && fabs(p->grad[0] + 1.0/3.0) < 1e-10);
        printf("reg losses:    %s\n", ok ? "PASS" : "FAIL");
        triad_tensor_free(p); triad_tensor_free(t);
        triad_tensor_free(l1); triad_tensor_free(hub);
    }

    /* ── Test 11: Linear layer forward ── */
    {
        TriadLinear *lin = triad_linear_new(2, 3, 1);
        int32_t xs[] = {4, 2};
        TriadTensor *x = triad_tensor_ones(2, xs, 1);
        TriadTensor *y = triad_linear_forward(lin, x);
        int ok = (y->ndim == 2 && y->shape[0] == 4 && y->shape[1] == 3);
        printf("linear fwd:    %s (shape=%dx%d)\n", ok ? "PASS" : "FAIL",
               y->shape[0], y->shape[1]);
        triad_tensor_free(x); triad_tensor_free(y);
        triad_linear_free(lin);
    }

    /* ── Test 12: native embedding + layer norm + 3D linear ── */
    {
        TriadEmbedding *emb = triad_embedding_new(8, 4);
        TriadLayerNorm *ln = triad_layer_norm_new(4, 1e-5);
        TriadLinear *lin = triad_linear_new(4, 6, 1);

        int32_t ids_shape[] = {2, 3};
        TriadTensor *ids = triad_tensor_from_data(2, ids_shape,
            (double[]){1, 2, 3, 3, 2, 1}, 0);
        TriadTensor *x = triad_embedding_forward(emb, ids);
        TriadTensor *n = triad_layer_norm_forward(ln, x);
        TriadTensor *y = triad_linear_forward(lin, n);
        TriadTensor *loss = triad_tensor_sum(y);
        triad_tensor_backward(loss, NULL);

        int ok = (x && x->ndim == 3 && x->shape[0] == 2 && x->shape[1] == 3 &&
                  x->shape[2] == 4 &&
                  n && y && y->ndim == 3 && y->shape[0] == 2 &&
                  y->shape[1] == 3 && y->shape[2] == 6 &&
                  emb->weight->grad && ln->gamma->grad && lin->weight->grad);
        printf("embed/ln/lin3d:%s\n", ok ? " PASS" : " FAIL");

        triad_tensor_free(ids); triad_tensor_free(x);
        triad_tensor_free(n); triad_tensor_free(y);
        triad_tensor_free(loss);
        triad_embedding_free(emb);
        triad_layer_norm_free(ln);
        triad_linear_free(lin);
    }

    /* ── Test 13: Sequential with embedding + layer norm ── */
    {
        TriadSequential *seq = triad_sequential_new(4);
        triad_sequential_set(seq, 0, TRIAD_LAYER_EMBEDDING, triad_embedding_new(12, 5));
        triad_sequential_set(seq, 1, TRIAD_LAYER_LAYER_NORM, triad_layer_norm_new(5, 1e-5));
        triad_sequential_set(seq, 2, TRIAD_LAYER_LINEAR, triad_linear_new(5, 7, 1));
        triad_sequential_set(seq, 3, TRIAD_LAYER_FLATTEN, NULL);

        int32_t ids_shape[] = {2, 2};
        TriadTensor *ids = triad_tensor_from_data(2, ids_shape, (double[]){1, 4, 4, 1}, 0);
        TriadTensor *out = triad_sequential_forward(seq, ids);
        TriadTensor *params[16];
        int32_t np = triad_sequential_params(seq, params, 16);
        int ok = (out && out->ndim == 2 && out->shape[0] == 2 &&
                  out->shape[1] == 14 && np == 5);
        printf("seq ml blocks: %s\n", ok ? "PASS" : "FAIL");

        triad_tensor_free(ids);
        triad_tensor_free(out);
        triad_sequential_free(seq);
    }

    /* ── Test 14: XOR training with Sequential + Adam ── */
    {
        printf("\n=== XOR Training ===\n");

        /* Build: Linear(2,8) -> ReLU -> Linear(8,1) -> Sigmoid */
        TriadSequential *model = triad_sequential_new(4);
        triad_sequential_set(model, 0, TRIAD_LAYER_LINEAR, triad_linear_new(2, 8, 1));
        triad_sequential_set(model, 1, TRIAD_LAYER_RELU, NULL);
        triad_sequential_set(model, 2, TRIAD_LAYER_LINEAR, triad_linear_new(8, 1, 1));
        triad_sequential_set(model, 3, TRIAD_LAYER_SIGMOID, NULL);

        /* Collect params */
        TriadTensor *params[64];
        int32_t np = triad_sequential_params(model, params, 64);
        printf("params: %d tensors\n", np);

        /* Adam optimizer */
        TriadAdam *opt = triad_adam_new(params, np, 0.01, 0.9, 0.999, 1e-8);

        /* XOR data */
        double xor_x[4][2] = {{0,0},{0,1},{1,0},{1,1}};
        double xor_y[4] = {0, 1, 1, 0};

        double last_loss = 999;
        for (int epoch = 0; epoch < 2000; epoch++) {
            double epoch_loss = 0;
            for (int s = 0; s < 4; s++) {
                triad_adam_zero_grad(opt);

                /* Forward */
                int32_t xs[] = {2};
                TriadTensor *inp = triad_tensor_from_data(1, xs, xor_x[s], 0);
                TriadTensor *pred = triad_sequential_forward(model, inp);

                /* MSE loss manually: (pred - target)^2 */
                int32_t ts[] = {1};
                TriadTensor *tgt = triad_tensor_from_data(1, ts, &xor_y[s], 0);
                TriadTensor *loss = triad_tensor_mse_loss(pred, tgt);

                epoch_loss += loss->data[0];

                /* Backward + step */
                triad_tensor_backward(loss, NULL);
                triad_adam_step(opt);

                triad_tensor_free(inp); triad_tensor_free(tgt);
                /* pred, loss are intermediate — freed by caller in production,
                   leak here for simplicity in test */
            }
            last_loss = epoch_loss / 4.0;
            if ((epoch + 1) % 500 == 0)
                printf("epoch %4d: loss=%.6f\n", epoch + 1, last_loss);
        }

        /* Evaluate */
        printf("\nPredictions:\n");
        int correct = 0;
        for (int s = 0; s < 4; s++) {
            int32_t xs[] = {2};
            TriadTensor *inp = triad_tensor_from_data(1, xs, xor_x[s], 0);
            TriadTensor *pred = triad_sequential_forward(model, inp);
            double p = pred->data[0];
            int label = p > 0.5 ? 1 : 0;
            int target = (int)xor_y[s];
            printf("  [%.0f, %.0f] -> %.4f (pred=%d, target=%d) %s\n",
                   xor_x[s][0], xor_x[s][1], p, label, target,
                   label == target ? "OK" : "WRONG");
            if (label == target) correct++;
            triad_tensor_free(inp);
        }
        printf("\nXOR accuracy: %d/4\n", correct);
        printf("XOR training: %s\n", correct == 4 ? "PASS" : "FAIL");

        triad_adam_free(opt);
        triad_sequential_free(model);
    }

    return 0;
}
