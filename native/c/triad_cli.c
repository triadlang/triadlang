#include "triad_check.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
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

static void emit_diag(FILE *out, const TriadDiag *d) {
    fprintf(out, "error[%s]: %s",
            d->kind ? d->kind : "?",
            d->msg  ? d->msg  : "");
    if (d->file && d->file[0]) fprintf(out, "\n  file: %s", d->file);
    if (d->line)               fprintf(out, "\n  line: %d", d->line);
    if (d->col)                fprintf(out, "\n  col: %d",  d->col);
    fputc('\n', out);
}

static int cmd_check(int argc, char **argv) {

    if (argc < 2) {
        fprintf(stderr, "usage: triad check <file.tri>\n");
        return 1;
    }
    const char *path = argv[1];
    if (!file_exists(path)) {
        fprintf(stderr, "error: file not found: %s\n", path);
        return 1;
    }

    char *src = slurp(path);
    if (!src) {
        fprintf(stderr, "error: cannot read: %s\n", path);
        return 1;
    }

    TriadArena arena;
    triad_arena_init(&arena, 1 << 16);

    TriadDiag diag = {0};
    TriadAstNode *mod = triad_parse_source(&arena, src, path, &diag);
    if (!mod) {
        emit_diag(stderr, &diag);
        free(src);
        triad_arena_free(&arena);
        return 1;
    }

    TriadCheckErrors errs = {0};
    int rc = triad_check_module(&arena, mod, &errs);
    if (rc != 0) {
        for (size_t k = 0; k < errs.len; ++k) {
            fputs(errs.items[k], stderr);
            if (k + 1 < errs.len) fputc('\n', stderr);
        }
        fputc('\n', stderr);
        free(src);
        triad_arena_free(&arena);
        return 1;
    }

    printf("check: %s OK\n", path);
    free(src);
    triad_arena_free(&arena);
    return 0;
}

static int usage(void) {
    fprintf(stderr,
        "usage: triad <command> [args...]\n"
        "  check <file.tri>   Lex + parse + type-check a TriadLang source file.\n");
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) return usage();
    const char *cmd = argv[1];
    if (strcmp(cmd, "check") == 0) return cmd_check(argc - 1, argv + 1);
    if (strcmp(cmd, "-h") == 0 || strcmp(cmd, "--help") == 0) { usage(); return 0; }
    fprintf(stderr, "triad: unknown command '%s'\n", cmd);
    return usage();
}
