/* test_llm_native.c — Test LLM runtime: tokenizer + transformer + sampling */
#include "triad_llm.h"
#include "triad_rt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    printf("=== TriadLang LLM Native Tests ===\n\n");

    /* ── Test 1: Tokenizer basics ── */
    {
        TriadTokenizer *tok = triad_tokenizer_new();
        triad_tokenizer_add_token(tok, 0, "<pad>");
        triad_tokenizer_add_token(tok, 1, "<s>");
        triad_tokenizer_add_token(tok, 2, "</s>");
        triad_tokenizer_add_token(tok, 3, "h");
        triad_tokenizer_add_token(tok, 4, "e");
        triad_tokenizer_add_token(tok, 5, "l");
        triad_tokenizer_add_token(tok, 6, "o");
        triad_tokenizer_add_token(tok, 7, " ");
        triad_tokenizer_add_token(tok, 8, "he");
        triad_tokenizer_add_token(tok, 9, "ll");
        triad_tokenizer_add_token(tok, 10, "lo");
        triad_tokenizer_add_token(tok, 11, "hel");
        triad_tokenizer_add_token(tok, 12, "hello");

        /* Add merges */
        triad_tokenizer_add_merge(tok, "h", "e", 0);
        triad_tokenizer_add_merge(tok, "l", "l", 1);
        triad_tokenizer_add_merge(tok, "he", "ll", 2);
        triad_tokenizer_add_merge(tok, "hel", "l", 3);
        triad_tokenizer_add_merge(tok, "hell", "o", 4);

        int32_t ids[64];
        int32_t n = triad_tokenizer_encode(tok, "hello", ids, 64);

        printf("tokenizer encode 'hello': %d tokens [", n);
        for (int32_t i = 0; i < n; i++) {
            if (i > 0) printf(", ");
            printf("%d('%s')", ids[i], triad_tokenizer_id_to_str(tok, ids[i]));
        }
        printf("]\n");

        char *decoded = triad_tokenizer_decode(tok, ids, n);
        printf("tokenizer decode: '%s'\n", decoded);
        int ok = (strcmp(decoded, "hello") == 0);
        printf("tokenizer: %s\n\n", ok ? "PASS" : "FAIL");
        free(decoded);
        triad_tokenizer_free(tok);
    }

    /* ── Test 2: Sampling ── */
    {
        double logits[] = {1.0, 5.0, 2.0, 0.5, 3.0};
        int32_t greedy = triad_sample_greedy(logits, 5);
        printf("greedy sample: %d (expected 1)\n", greedy);
        printf("greedy: %s\n", greedy == 1 ? "PASS" : "FAIL");

        uint64_t rng = 42;
        int32_t topk = triad_sample_top_k(logits, 5, 2, 1.0, &rng);
        printf("top-k=2 sample: %d (should be 1 or 4)\n", topk);
        printf("top-k: %s\n\n", (topk == 1 || topk == 4) ? "PASS" : "FAIL");
    }

    /* ── Test 3: KV Cache ── */
    {
        TriadKVCache *kv = triad_kv_cache_new(16, 2, 4, 1);
        double k_data[] = {1,2,3,4, 5,6,7,8};  /* 2 heads * 4 head_dim */
        double v_data[] = {0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8};
        triad_kv_cache_write(kv, 0, k_data, v_data);
        const double *cached_k = triad_kv_cache_get_k(kv, 0);
        int ok = (cached_k[0] == 1.0 && cached_k[7] == 8.0);
        printf("kv cache write/read: %s\n", ok ? "PASS" : "FAIL");
        triad_kv_cache_advance(kv);
        printf("kv cache pos: %d (expected 1)\n", kv->pos);
        printf("kv cache: %s\n\n", kv->pos == 1 ? "PASS" : "FAIL");
        triad_kv_cache_free(kv);
    }

    /* ── Test 4: RMSNorm ── */
    {
        int32_t s[] = {4};
        TriadTensor *x = triad_tensor_from_data(1, s, (double[]){1,2,3,4}, 0);
        TriadTensor *w = triad_tensor_ones(1, s, 0);
        TriadTensor *out = triad_rms_norm(x, w, 1e-5);
        /* rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386 */
        /* out[0] = 1/2.7386 ≈ 0.3651 */
        int ok = (fabs(out->data[0] - 1.0/sqrt(7.5)) < 0.01);
        printf("rms_norm: %s (out[0]=%.4f, expected %.4f)\n",
               ok ? "PASS" : "FAIL", out->data[0], 1.0/sqrt(7.5));
        triad_tensor_free(x); triad_tensor_free(w); triad_tensor_free(out);
    }

    /* ── Test 5: RoPE ── */
    {
        double q[] = {1,0, 1,0};
        double k[] = {1,0, 1,0};
        triad_rope(q, k, 4, 0, 10000.0);
        /* pos=0: angle=0 for all dims, so cos=1, sin=0 -> no change */
        int ok = (fabs(q[0] - 1.0) < 1e-6 && fabs(q[1] - 0.0) < 1e-6);
        printf("rope pos=0: %s\n", ok ? "PASS" : "FAIL");

        double q2[] = {1,0, 1,0};
        double k2[] = {1,0, 1,0};
        triad_rope(q2, k2, 4, 1, 10000.0);
        /* pos=1: should rotate */
        ok = (fabs(q2[0] - 1.0) > 1e-6 || fabs(q2[1] - 0.0) > 1e-6);
        printf("rope pos=1: %s (rotated: q[0]=%.4f q[1]=%.4f)\n\n",
               ok ? "PASS" : "FAIL", q2[0], q2[1]);
    }

    /* ── Test 6: Mini transformer forward pass ── */
    {
        TriadLLMConfig cfg = {
            .dim = 8,
            .hidden_dim = 16,
            .n_layers = 1,
            .n_heads = 2,
            .vocab_size = 10,
            .max_seq_len = 32,
            .norm_eps = 1e-5,
            .rope_theta = 10000.0
        };
        TriadLLM *model = triad_llm_new(&cfg);

        /* Init with small random weights */
        triad_ml_seed(42);
        for (int64_t i = 0; i < model->tok_emb->size; i++)
            model->tok_emb->data[i] = 0.1 * (((double)(i % 17) - 8) / 8.0);
        for (int32_t l = 0; l < 1; l++) {
            TriadTransformerLayer *lay = &model->layers[l];
            for (int64_t i = 0; i < lay->wq->size; i++) {
                double v = 0.01 * (((double)(i % 13) - 6) / 6.0);
                lay->wq->data[i] = v;
                lay->wk->data[i] = v * 0.8;
                lay->wv->data[i] = v * 0.5;
                lay->wo->data[i] = v * 0.3;
            }
            for (int64_t i = 0; i < lay->w_gate->size; i++) {
                lay->w_gate->data[i] = 0.01 * (((double)(i % 11) - 5) / 5.0);
                lay->w_up->data[i] = 0.01 * (((double)(i % 7) - 3) / 3.0);
            }
            for (int64_t i = 0; i < lay->w_down->size; i++)
                lay->w_down->data[i] = 0.01 * (((double)(i % 9) - 4) / 4.0);
        }
        for (int64_t i = 0; i < model->output->size; i++)
            model->output->data[i] = 0.01 * (((double)(i % 19) - 9) / 9.0);

        /* Forward pass */
        TriadTensor *logits = triad_llm_forward(model, 3, 0);
        printf("mini transformer forward:\n");
        printf("  logits shape: %d (expected %d)\n", (int)logits->size, cfg.vocab_size);
        int ok = ((int)logits->size == cfg.vocab_size);

        /* Check logits aren't all zero */
        double sum = 0;
        for (int32_t i = 0; i < cfg.vocab_size; i++) sum += fabs(logits->data[i]);
        ok = ok && (sum > 0);
        printf("  logits sum(|x|): %.6f (should be > 0)\n", sum);

        /* Sample greedy */
        int32_t next = triad_sample_greedy(logits->data, cfg.vocab_size);
        printf("  greedy next token: %d\n", next);
        printf("mini transformer: %s\n\n", ok ? "PASS" : "FAIL");

        /* Second token (autoregressive) */
        TriadTensor *logits2 = triad_llm_forward(model, next, 1);
        int32_t next2 = triad_sample_greedy(logits2->data, cfg.vocab_size);
        printf("  autoregressive token 2: %d\n", next2);
        printf("  kv_cache pos: %d (expected 2)\n", model->kv_cache->pos);
        printf("autoregressive: %s\n", model->kv_cache->pos == 2 ? "PASS" : "FAIL");

        triad_tensor_free(logits);
        triad_tensor_free(logits2);
        triad_llm_free(model);
    }

    return 0;
}
