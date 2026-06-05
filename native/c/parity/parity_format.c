/* parity_format.c — Parse a .tri file, run native formatter, print to
 * stdout. Compared by parity_run_all.py against scripts/format_to_text.py.
 */
#include "triad_format.h"

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
    if (argc < 2) {
        fprintf(stderr, "usage: parity_format <file.tri>\n");
        return 2;
    }
    char *src = slurp(argv[1]);
    if (!src) return 1;

    TriadArena arena;
    triad_arena_init(&arena, 1 << 16);

    TriadDiag diag = {0};
    TriadAstNode *mod = triad_parse_source(&arena, src, argv[1], &diag);
    if (!mod) {
        fprintf(stderr, "error[%s]: %s @ L%d:C%d\n",
                diag.kind ? diag.kind : "?",
                diag.msg ? diag.msg : "?",
                diag.line, diag.col);
        free(src); triad_arena_free(&arena); return 1;
    }
    const char *out = triad_format_module(&arena, mod);
    if (out) fputs(out, stdout);
    free(src);
    triad_arena_free(&arena);
    return 0;
}
