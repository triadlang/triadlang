/* parity_check.c — Parse + run native typecheck, print errors. Exit 1
 * on failure, 0 on success. Output format matches Python `\n`-joined
 * TypeCheckError text. */
#include "triad_check.h"

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

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: parity_check <file.tri>\n"); return 2; }
    char *src = slurp(argv[1]);
    if (!src) return 1;

    TriadArena arena;
    triad_arena_init(&arena, 1 << 16);

    TriadDiag diag = {0};
    TriadAstNode *mod = triad_parse_source(&arena, src, argv[1], &diag);
    if (!mod) {
        fprintf(stderr, "error[%s]: %s\n  file: %s\n  line: %d\n  col: %d\n",
                diag.kind ? diag.kind : "?", diag.msg ? diag.msg : "?",
                diag.file ? diag.file : "?", diag.line, diag.col);
        free(src); triad_arena_free(&arena); return 1;
    }

    TriadCheckErrors errs = {0};
    int rc = triad_check_module(&arena, mod, &errs);
    for (size_t k = 0; k < errs.len; ++k) {
        fputs(errs.items[k], stdout);
        if (k + 1 < errs.len) fputc('\n', stdout);
    }
    if (errs.len) fputc('\n', stdout);
    free(src);
    triad_arena_free(&arena);
    return rc == 0 ? 0 : 1;
}
