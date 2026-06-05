/* parity_lsp.c — Run a single LSP handler invocation and print its
 * JSON result (or "null"). Mirrors scripts/lsp_to_text.py.
 *
 * Usage:
 *   parity_lsp initialize
 *   parity_lsp documentSymbol <file.tri> <uri>
 *   parity_lsp completion     <file.tri> <line> <char>
 *   parity_lsp definition     <file.tri> <line> <char> <uri>
 *   parity_lsp hover          <file.tri> <line> <char>
 */
#include "triad_lsp.h"

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
        fprintf(stderr, "usage: parity_lsp <command> [args...]\n");
        return 2;
    }
    TriadArena arena;
    triad_arena_init(&arena, 1 << 16);

    const char *cmd = argv[1];
    const char *out = NULL;

    if (strcmp(cmd, "initialize") == 0) {
        out = triad_lsp_initialize_result_json(&arena);
    } else if (strcmp(cmd, "documentSymbol") == 0 && argc >= 4) {
        char *src = slurp(argv[2]);
        if (src) {
            out = triad_lsp_document_symbols_json(&arena, src, argv[3]);
            free(src);
        }
    } else if (strcmp(cmd, "completion") == 0 && argc >= 5) {
        char *src = slurp(argv[2]);
        if (src) {
            out = triad_lsp_completions_json(&arena, src,
                                             atoi(argv[3]), atoi(argv[4]));
            free(src);
        }
    } else if (strcmp(cmd, "definition") == 0 && argc >= 6) {
        char *src = slurp(argv[2]);
        if (src) {
            out = triad_lsp_definition_json(&arena, src,
                                            atoi(argv[3]), atoi(argv[4]),
                                            argv[5]);
            free(src);
        }
    } else if (strcmp(cmd, "hover") == 0 && argc >= 5) {
        char *src = slurp(argv[2]);
        if (src) {
            out = triad_lsp_hover_json(&arena, src,
                                       atoi(argv[3]), atoi(argv[4]));
            free(src);
        }
    } else {
        fprintf(stderr, "unknown command or wrong args: %s\n", cmd);
        triad_arena_free(&arena);
        return 2;
    }

    if (out) printf("%s\n", out);
    else     printf("null\n");
    triad_arena_free(&arena);
    return 0;
}
