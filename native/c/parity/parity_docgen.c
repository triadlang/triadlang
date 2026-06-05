/* parity_docgen.c — Run native docgen, print Markdown or HTML.
 * Usage:
 *   parity_docgen <file.tri> [--html]
 */
#define _POSIX_C_SOURCE 200809L
#include "triad_docgen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>

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
        fprintf(stderr, "usage: parity_docgen <file.tri> [--html]\n");
        return 2;
    }
    int as_html = 0;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--html") == 0) as_html = 1;
    }
    char *src = slurp(argv[1]);
    if (!src) return 1;

    /* The Python CLI passes os.path.basename(path) as filename. Mirror. */
    char *copy = strdup(argv[1]);
    const char *fname = basename(copy);

    TriadArena arena;
    triad_arena_init(&arena, 1 << 16);

    const char *out;
    if (as_html) out = triad_docgen_html(&arena, src, fname);
    else         out = triad_docgen_markdown(&arena, src, fname);

    if (out) fputs(out, stdout);
    free(copy);
    free(src);
    triad_arena_free(&arena);
    return 0;
}
