/* parity_ir.c — Read a .tri file, parse + lower, dump IR JSON. */
#include "triad_ir.h"

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
    if (argc < 2) { fprintf(stderr, "usage: parity_ir <file.tri>\n"); return 2; }
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
    TriadIRNode *ir = triad_ir_lower_module(&arena, mod, &diag);
    if (!ir) { fprintf(stderr, "lower failed\n"); free(src); triad_arena_free(&arena); return 1; }
    char *json = triad_ir_dump_json(ir, 2);
    fputs(json, stdout);
    fputc('\n', stdout);
    free(json);
    free(src);
    triad_arena_free(&arena);
    return 0;
}
