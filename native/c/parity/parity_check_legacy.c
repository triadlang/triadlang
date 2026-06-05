/* parity_check_legacy.c — Parse + run native legacy typecheck, print
 * errors. Exit 1 on failure, 0 on success. Output format matches Python
 * `\n`-joined TypeCheckError text.
 *
 * Usage:
 *   parity_check_legacy <file.tri> [--regimes <path>]
 *
 * The optional --regimes file contains one regime name per line. If
 * omitted, an empty regime registry is used (which makes any regime
 * reference unknown).
 */
#include "triad_check_legacy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    buf[n] = '\0';
    fclose(f);
    return buf;
}

static int load_regimes(const char *path, char ***out, size_t *out_n) {
    char *raw = slurp(path);
    if (!raw) return -1;
    size_t cap = 32, n = 0;
    char **arr = (char **)malloc(sizeof(char *) * cap);
    char *p = raw;
    while (*p) {
        char *line = p;
        while (*p && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        while (len && (line[len - 1] == '\r' || line[len - 1] == ' ')) len--;
        if (len > 0) {
            if (n + 1 > cap) { cap *= 2; arr = (char **)realloc(arr, sizeof(char *) * cap); }
            char *s = (char *)malloc(len + 1);
            memcpy(s, line, len);
            s[len] = '\0';
            arr[n++] = s;
        }
        if (*p == '\n') p++;
    }
    free(raw);
    *out = arr;
    *out_n = n;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: parity_check_legacy <file.tri> [--regimes <path>]\n");
        return 2;
    }
    const char *file = argv[1];
    const char *regimes_path = NULL;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--regimes") == 0 && i + 1 < argc) {
            regimes_path = argv[++i];
        }
    }

    char *src = slurp(file);
    if (!src) return 1;

    char **reg_names = NULL;
    size_t reg_count = 0;
    if (regimes_path) {
        if (load_regimes(regimes_path, &reg_names, &reg_count) != 0) {
            fprintf(stderr, "failed to load regimes file: %s\n", regimes_path);
            free(src);
            return 1;
        }
    }

    TriadArena arena;
    triad_arena_init(&arena, 1 << 16);

    TriadDiag diag = {0};
    TriadLegacyNode *prog = triad_legacy_parse_source(&arena, src, &diag);
    if (!prog) {
        fprintf(stderr, "error[%s]: %s @ L%d:C%d\n",
                diag.kind ? diag.kind : "?",
                diag.msg ? diag.msg : "?",
                diag.line, diag.col);
        free(src);
        triad_arena_free(&arena);
        for (size_t i = 0; i < reg_count; ++i) free(reg_names[i]);
        free(reg_names);
        return 1;
    }

    TriadCheckLegacyErrors errs = {0};
    int rc = triad_check_legacy_program(&arena, prog,
                                        (const char *const *)reg_names,
                                        reg_count, &errs);
    for (size_t k = 0; k < errs.len; ++k) {
        fputs(errs.items[k], stdout);
        if (k + 1 < errs.len) fputc('\n', stdout);
    }
    if (errs.len) fputc('\n', stdout);

    free(src);
    triad_arena_free(&arena);
    for (size_t i = 0; i < reg_count; ++i) free(reg_names[i]);
    free(reg_names);
    return rc == 0 ? 0 : 1;
}
