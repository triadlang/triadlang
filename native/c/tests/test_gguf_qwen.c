/* test_gguf_qwen.c — Read Qwen3 27B GGUF and print model info */
#include "triad_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = "docs/Qwen3.6-27B-Q4_K_M.gguf";
    if (argc > 1) path = argv[1];

    printf("Opening %s...\n", path);
    GGUFFile *f = gguf_open(path);
    if (!f) { fprintf(stderr, "failed to open\n"); return 1; }

    printf("GGUF v%d\n", f->version);
    printf("KV pairs: %ld\n", (long)f->n_kv);
    printf("Tensors:  %ld\n", (long)f->n_tensors);
    printf("Data offset: %lu\n", (unsigned long)f->data_offset);
    printf("\n");

    /* Print model config */
    const char *keys[] = {
        "general.architecture", "general.name",
        "qwen35.block_count", "qwen35.embedding_length",
        "qwen35.feed_forward_length", "qwen35.attention.head_count",
        "qwen35.attention.head_count_kv", "qwen35.attention.layer_norm_rms_epsilon",
        "qwen35.context_length", "qwen35.rope.freq_base",
        "tokenizer.ggml.model", "tokenizer.ggml.tokens",
        NULL
    };

    printf("=== Model Config ===\n");
    for (int i = 0; keys[i]; i++) {
        const GGUFKeyValue *kv = gguf_find_kv(f, keys[i]);
        if (!kv) {
            printf("  %-50s = (not found)\n", keys[i]);
            continue;
        }
        switch (kv->type) {
            case GGUF_TYPE_UINT32: printf("  %-50s = %u\n", keys[i], kv->val.u32); break;
            case GGUF_TYPE_INT32:  printf("  %-50s = %d\n", keys[i], kv->val.i32); break;
            case GGUF_TYPE_UINT64: printf("  %-50s = %lu\n", keys[i], (unsigned long)kv->val.u64); break;
            case GGUF_TYPE_FLOAT32:printf("  %-50s = %g\n", keys[i], kv->val.f32); break;
            case GGUF_TYPE_STRING: printf("  %-50s = %s\n", keys[i], kv->val.str); break;
            case GGUF_TYPE_ARRAY:  printf("  %-50s = array[%lu]\n", keys[i], (unsigned long)kv->val.arr.len); break;
            default: printf("  %-50s = (type %d)\n", keys[i], kv->type); break;
        }
    }

    /* Print first 20 tensors */
    printf("\n=== Tensors (first 20) ===\n");
    const char *type_names[] = {
        [0]="F32", [1]="F16", [2]="Q4_0", [3]="Q4_1",
        [6]="Q5_0", [7]="Q5_1", [8]="Q8_0",
        [10]="Q2_K", [11]="Q3_K", [12]="Q4_K", [13]="Q5_K",
        [14]="Q6_K", [15]="Q8_K", [30]="BF16"
    };
    for (int64_t i = 0; i < f->n_tensors && i < 20; i++) {
        GGUFTensorInfo *t = &f->tensors[i];
        const char *tn = (t->type < 31) ? type_names[t->type] : "?";
        if (!tn) tn = "?";
        printf("  %-60s [", t->name);
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (d > 0) printf(", ");
            printf("%ld", (long)t->shape[d]);
        }
        printf("] %s (%ld elems, %.2f MB)\n", tn, (long)t->n_elements,
               (double)t->n_bytes / (1024.0 * 1024.0));
    }

    /* Quick dequant test on first small tensor */
    printf("\n=== Dequant Test ===\n");
    for (int64_t i = 0; i < f->n_tensors; i++) {
        GGUFTensorInfo *t = &f->tensors[i];
        if (t->n_elements <= 4096 && t->n_elements > 0) {
            printf("Dequantizing %s (%ld elems, type %d)...\n",
                   t->name, (long)t->n_elements, t->type);
            double *data = gguf_dequantize(f, t);
            printf("  first 8 values: ");
            for (int j = 0; j < 8 && j < t->n_elements; j++)
                printf("%.6f ", data[j]);
            printf("\n");

            /* Stats */
            double mn = data[0], mx = data[0], sum = 0;
            for (int64_t j = 0; j < t->n_elements; j++) {
                if (data[j] < mn) mn = data[j];
                if (data[j] > mx) mx = data[j];
                sum += data[j];
            }
            printf("  min=%.6f max=%.6f mean=%.6f\n", mn, mx, sum / t->n_elements);
            free(data);
            break;
        }
    }

    gguf_close(f);
    printf("\nDone.\n");
    return 0;
}
