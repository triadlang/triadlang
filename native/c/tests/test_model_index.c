#include "triad_model_index.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int check_tensor(TriadModelIndex *idx, const char *name) {
    TriadSafeTensorFile *file = NULL;
    const TriadSafeTensorInfo *t = triad_model_find_tensor(idx, name, &file);
    if (!t || !file) {
        printf("missing: %s\n", name);
        return 0;
    }
    printf("found: %s shard=%s dtype=%s shape=[", name, file->path,
           triad_safetensors_dtype_name(t->dtype));
    for (int32_t i = 0; i < t->ndim; i++) {
        if (i) printf(",");
        printf("%lld", (long long)t->shape[i]);
    }
    printf("]\n");

    if (t->ndim == 2) {
        double *row = malloc((size_t)t->shape[1] * sizeof(double));
        if (!row) return 0;
        int got = triad_model_read_row_f64(idx, name, 0, row, t->shape[1]);
        double l2 = 0.0;
        for (int i = 0; got > 0 && i < got; i++) l2 += row[i] * row[i];
        printf("  row0: n=%d l2=%.6g\n", got, sqrt(l2));
        free(row);
        return got > 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "../../models";
    TriadModelIndex idx;
    if (triad_model_index_load(&idx, dir) != 0) {
        fprintf(stderr, "failed to load model index: %s\n", dir);
        return 1;
    }
    printf("entries=%d\n", idx.nentries);

    const char *need[] = {
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        NULL
    };
    int ok = 1;
    for (int i = 0; need[i]; i++) ok = check_tensor(&idx, need[i]) && ok;
    triad_model_index_free(&idx);
    printf("model index: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
