#include "triad_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void stats(const char *label, const double *x, int64_t n) {
    double mn = x[0], mx = x[0], sum = 0.0, l2 = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (x[i] < mn) mn = x[i];
        if (x[i] > mx) mx = x[i];
        sum += x[i];
        l2 += x[i] * x[i];
    }
    printf("%s: n=%lld min=%.6g max=%.6g mean=%.6g l2=%.6g\n",
           label, (long long)n, mn, mx, sum / (double)n, sqrt(l2));
}

static void describe_tensor(const TriadSafeTensorInfo *t) {
    printf("  %s %s [", t->name, triad_safetensors_dtype_name(t->dtype));
    for (int32_t i = 0; i < t->ndim; i++) {
        if (i) printf(",");
        printf("%lld", (long long)t->shape[i]);
    }
    printf("] offsets=[%llu,%llu]\n",
           (unsigned long long)t->data_begin,
           (unsigned long long)t->data_end);
}

int main(int argc, char **argv) {
    const char *arg = argc > 1 ? argv[1] : "../../models";
    char path[4096];
    if (strstr(arg, ".safetensors")) {
        snprintf(path, sizeof(path), "%s", arg);
    } else {
        snprintf(path, sizeof(path), "%s/model.safetensors-00001-of-00004.safetensors", arg);
    }

    TriadSafeTensorFile f;
    if (triad_safetensors_open(path, &f) != 0) {
        fprintf(stderr, "failed to open safetensors shard: %s\n", path);
        return 1;
    }

    printf("file: %s\n", path);
    printf("size=%llu header=%llu tensors=%d\n",
           (unsigned long long)f.size,
           (unsigned long long)f.header_len,
           f.ntensors);
    for (int32_t i = 0; i < f.ntensors && i < 8; i++) describe_tensor(&f.tensors[i]);

    const char *names[] = {
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        NULL
    };
    for (int i = 0; names[i]; i++) {
        const TriadSafeTensorInfo *t = triad_safetensors_find(&f, names[i]);
        if (!t) {
            printf("missing: %s\n", names[i]);
            continue;
        }
        describe_tensor(t);
        double *row = malloc((size_t)t->shape[1] * sizeof(double));
        if (!row) {
            triad_safetensors_close(&f);
            return 1;
        }
        int got = triad_safetensors_read_row_f64(&f, t, 0, row, t->shape[1]);
        if (got < 0) {
            printf("row read failed: %s\n", names[i]);
            free(row);
            continue;
        }
        stats(names[i], row, got);
        free(row);
    }

    triad_safetensors_close(&f);
    return 0;
}
