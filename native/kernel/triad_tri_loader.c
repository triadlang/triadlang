#include "triad_tri_loader.h"
#include "triad_vfs.h"
#include "triad_mm.h"
#include "triad_serial.h"
#include <string.h>

static const char *KEYWORDS[] = {
    "let", "const", "fn", "return", "if", "else", "elif", "for", "in",
    "while", "break", "continue", "type", "class", "super", "import",
    "from", "as", "true", "false", "none", "and", "or", "not", "try",
    "catch", "finally", "throw", "self", "match", "case", "yield",
    "async", "await", "with", "reg", "entity", "world", "couple",
    "pair", "ring", "observe", "run", "evolve", "sequence", "substrate",
    "via", "each_for", "composed_of", "assert", "pass", "del",
    "is", "inherits", "persistent", "extended", "structurally_open",
    "mem_memory", "atomic", "anti_collapsed", "over_seeds",
    "capability",
    NULL
};

static int is_keyword(const char *s) {
    for (int i = 0; KEYWORDS[i]; i++) {
        int k = 0;
        while (KEYWORDS[i][k] && s[k] == KEYWORDS[i][k]) k++;
        if (KEYWORDS[i][k] == 0 && s[k] == 0) return 1;
    }
    return 0;
}

void triad_tri_lex_init(TriLexer *lex, const char *src) {
    lex->src = src;
    lex->pos = 0;
    lex->line = 1;
    lex->col = 1;
    lex->len = 0;
    while (src[lex->len]) lex->len++;
    lex->current.type = TRI_TOKEN_EOF;
    lex->current.value[0] = 0;
}

static int lex_skip_ws(TriLexer *lex) {
    while (lex->pos < lex->len) {
        char c = lex->src[lex->pos];
        if (c == '\n') {
            lex->line++;
            lex->col = 1;
            lex->pos++;
        } else if (c == ' ' || c == '\t' || c == '\r') {
            lex->col++;
            lex->pos++;
        } else if (c == '/' && lex->pos + 1 < lex->len && lex->src[lex->pos + 1] == '/') {
            while (lex->pos < lex->len && lex->src[lex->pos] != '\n') lex->pos++;
        } else if (c == '#') {
            while (lex->pos < lex->len && lex->src[lex->pos] != '\n') lex->pos++;
        } else if (c == '/' && lex->pos + 1 < lex->len && lex->src[lex->pos + 1] == '*') {
            lex->pos += 2;
            while (lex->pos < lex->len) {
                if (lex->src[lex->pos] == '*' && lex->pos + 1 < lex->len && lex->src[lex->pos + 1] == '/') {
                    lex->pos += 2;
                    break;
                }
                if (lex->src[lex->pos] == '\n') lex->line++;
                lex->pos++;
            }
        } else {
            break;
        }
    }
    return lex->pos >= lex->len;
}

int triad_tri_lex_next(TriLexer *lex, TriToken *out) {
    if (lex_skip_ws(lex)) {
        out->type = TRI_TOKEN_EOF;
        out->value[0] = 0;
        return 0;
    }

    char c = lex->src[lex->pos];
    out->line = lex->line;
    out->col = lex->col;

    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
        int n = 0;
        while (lex->pos < lex->len) {
            c = lex->src[lex->pos];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_') {
                if (n < 127) out->value[n] = c;
                n++;
                lex->pos++;
                lex->col++;
            } else break;
        }
        out->value[n] = 0;
        out->type = is_keyword(out->value) ? TRI_TOKEN_KEYWORD : TRI_TOKEN_IDENT;
        return 1;
    }

    if ((c >= '0' && c <= '9') || (c == '.' && lex->pos + 1 < lex->len && lex->src[lex->pos + 1] >= '0' && lex->src[lex->pos + 1] <= '9')) {
        int n = 0;
        while (lex->pos < lex->len) {
            c = lex->src[lex->pos];
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == 'x' || c == 'X' || c == 'b' || c == 'B') {
                if (n < 127) out->value[n] = c;
                n++;
                lex->pos++;
                lex->col++;
            } else break;
        }
        out->value[n] = 0;
        out->type = TRI_TOKEN_NUMBER;
        return 1;
    }

    if (c == '"') {
        lex->pos++;
        lex->col++;
        int n = 0;
        while (lex->pos < lex->len && lex->src[lex->pos] != '"') {
            if (lex->src[lex->pos] == '\\' && lex->pos + 1 < lex->len) {
                lex->pos++;
                lex->col++;
                char esc = lex->src[lex->pos];
                char actual = esc;
                if (esc == 'n') actual = '\n';
                else if (esc == 't') actual = '\t';
                else if (esc == '\\') actual = '\\';
                else if (esc == '"') actual = '"';
                if (n < 127) out->value[n] = actual;
                n++;
                lex->pos++;
                lex->col++;
            } else {
                if (n < 127) out->value[n] = lex->src[lex->pos];
                n++;
                lex->pos++;
                lex->col++;
            }
        }
        if (lex->pos < lex->len) { lex->pos++; lex->col++; }
        out->value[n] = 0;
        out->type = TRI_TOKEN_STRING;
        return 1;
    }

    if (c == 'f' && lex->pos + 1 < lex->len && lex->src[lex->pos + 1] == '"') {
        lex->pos += 2;
        lex->col += 2;
        int n = 0;
        while (lex->pos < lex->len && lex->src[lex->pos] != '"') {
            if (n < 127) out->value[n] = lex->src[lex->pos];
            n++;
            lex->pos++;
            lex->col++;
        }
        if (lex->pos < lex->len) { lex->pos++; lex->col++; }
        out->value[n] = 0;
        out->type = TRI_TOKEN_FSTRING;
        return 1;
    }

    out->value[0] = c;
    out->value[1] = 0;
    out->type = TRI_TOKEN_SYMBOL;
    lex->pos++;
    lex->col++;
    return 1;
}

int triad_tri_parse(const char *src, const char *path, TriModuleInfo *out) {
    memset(out, 0, sizeof(TriModuleInfo));
    int k = 0;
    while (path[k] && k < 255) { out->source_path[k] = path[k]; k++; }
    out->source_path[k] = 0;

    TriLexer lex;
    triad_tri_lex_init(&lex, src);
    TriToken tok;

    while (triad_tri_lex_next(&lex, &tok)) {
        if (tok.type == TRI_TOKEN_EOF) break;

        if (tok.type == TRI_TOKEN_KEYWORD) {
            if (strcmp(tok.value, "fn") == 0) out->n_functions++;
            else if (strcmp(tok.value, "import") == 0 || strcmp(tok.value, "from") == 0) out->n_imports++;
            else if (strcmp(tok.value, "reg") == 0) out->has_reg = true;
            else if (strcmp(tok.value, "observe") == 0 || strcmp(tok.value, "OBSERVE") == 0) out->has_observe = true;
            else if (strcmp(tok.value, "couple") == 0 || strcmp(tok.value, "pair") == 0 || strcmp(tok.value, "ring") == 0) out->has_couple = true;
            else if (strcmp(tok.value, "capability") == 0) {
                out->n_capabilities_declared++;
                triad_tri_lex_next(&lex, &tok);
                if (tok.type == TRI_TOKEN_IDENT) {
                    int n = 0;
                    while (out->caps_str[n]) n++;
                    if (n > 0 && n < 255) { out->caps_str[n] = ';'; n++; }
                    int j = 0;
                    while (tok.value[j] && n < 255) { out->caps_str[n] = tok.value[j]; n++; j++; }
                    out->caps_str[n] = 0;
                }
            }
        }
        out->n_statements++;
    }

    return 0;
}

int triad_tri_load_from_vfs(const char *vfs_path, TriModuleInfo *out) {
    int fd = triad_vfs_open(vfs_path, 0);
    if (fd < 0) {
        triad_serial_puts("[tri_loader] cannot open: ");
        triad_serial_puts(vfs_path);
        triad_serial_puts("\n");
        return -1;
    }

    char buf[4096];
    int n = triad_vfs_read(fd, buf, sizeof(buf) - 1);
    triad_vfs_close(fd);
    if (n < 0) return -1;
    buf[n] = 0;

    return triad_tri_parse(buf, vfs_path, out);
}