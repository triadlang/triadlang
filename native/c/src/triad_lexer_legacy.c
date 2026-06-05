/* triad_lexer_legacy.c — Port of frontend/lexer.py (v1 DSL). */
#include "triad_frontend.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const L_KEYWORDS[] = {
    "reg", "loop", "segment", "if", "else",
    "true", "false", "OUT", "HALT",
    "evolve", "for",
    "couple", "pair", "ring", "sequence", "via",
    "to", "kappa", "each_for",
    "OBSERVE", "over_seeds",
    "assert", "persistent", "extended", "structurally_open", "non_trivial_memory",
    "atomic", "anti_collapsed",
    "substrate", "composed_of",
    NULL
};

static const char *const L_OPCODES[] = {
    "MOV", "LOAD",
    "ADD", "SUB", "MUL", "DIV",
    "AND", "OR", "NOT", "XOR",
    "CMP",
    "INC", "DEC", "ZERO",
    "SHIFT_LEFT", "SHIFT_RIGHT",
    "PROBE",
    "CHECKPOINT",
    NULL
};

static const char *const L_LEGACY_KW[] = {
    "regime", "register", "init", "evolve", "mov", "for", "with",
    "kappa", "read", "let", "print", "none", "harmonic", "mode",
    NULL
};

static int in_set(const char *const *set, const char *s, size_t n) {
    for (size_t k = 0; set[k]; ++k) {
        const char *w = set[k];
        size_t wn = strlen(w);
        if (wn == n && memcmp(w, s, n) == 0) return 1;
    }
    return 0;
}

typedef struct {
    TriadArena         *arena;
    const char         *src;
    size_t              n;
    size_t              i;
    int                 line;
    int                 col;
    TriadDiag          *diag;

    TriadLegacyToken   *toks;
    size_t              toks_len;
    size_t              toks_cap;
} LL;

static void llerr(LL *L, int line, int col, const char *fmt, ...) {
    if (!L->diag) return;
    char buf[256];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
    L->diag->line = line;
    L->diag->col  = col;
    L->diag->file = "";
    L->diag->kind = "LEX";
    L->diag->msg  = triad_arena_strdup(L->arena, buf);
}

static int ll_push(LL *L, TriadLegacyToken t) {
    if (L->toks_len == L->toks_cap) {
        size_t nc = L->toks_cap ? L->toks_cap * 2 : 64;
        TriadLegacyToken *nt = (TriadLegacyToken *)realloc(L->toks, nc * sizeof(*nt));
        if (!nt) return -1;
        L->toks = nt; L->toks_cap = nc;
    }
    L->toks[L->toks_len++] = t;
    return 0;
}

static char ll_peek(LL *L, size_t off) {
    size_t p = L->i + off;
    return (p < L->n) ? L->src[p] : '\0';
}
static char ll_adv(LL *L) { char c = L->src[L->i++]; L->col++; return c; }

int triad_legacy_tokenize(TriadArena *arena, const char *src,
                          TriadLegacyTokenList *out, TriadDiag *diag) {
    LL L = {0};
    L.arena = arena;
    L.src   = src ? src : "";
    L.n     = strlen(L.src);
    L.line  = 1; L.col = 1;
    L.diag  = diag;
    if (diag) { diag->line = 0; diag->col = 0; diag->file = ""; diag->kind = NULL; diag->msg = NULL; }

    while (L.i < L.n) {
        char c = L.src[L.i];

        if (c == '\n') { ll_adv(&L); L.line++; L.col = 1; continue; }
        if (c == ' ' || c == '\t' || c == '\r') { ll_adv(&L); continue; }

        if (c == '/' && ll_peek(&L, 1) == '/') {
            while (L.i < L.n && L.src[L.i] != '\n') ll_adv(&L);
            continue;
        }
        if (c == '#') {
            while (L.i < L.n && L.src[L.i] != '\n') ll_adv(&L);
            continue;
        }

        /* annotation: @ident(...) — kept as single ANNOT with value "ident(args)" */
        if (c == '@') {
            int sl = L.line, sc = L.col;
            ll_adv(&L);
            size_t istart = L.i;
            while (L.i < L.n && (isalnum((unsigned char)L.src[L.i]) || L.src[L.i] == '_')) ll_adv(&L);
            size_t ilen = L.i - istart;
            const char *ident = triad_arena_strndup(arena, L.src + istart, ilen);
            const char *args  = "";
            if (L.i < L.n && L.src[L.i] == '(') {
                ll_adv(&L);
                int depth = 1;
                size_t astart = L.i;
                while (L.i < L.n && depth > 0) {
                    if (L.src[L.i] == '(') depth++;
                    else if (L.src[L.i] == ')') {
                        depth--;
                        if (depth == 0) break;
                    }
                    if (L.src[L.i] == '\n') { L.line++; L.col = 1; }
                    ll_adv(&L);
                }
                args = triad_arena_strndup(arena, L.src + astart, L.i - astart);
                if (L.i < L.n && L.src[L.i] == ')') ll_adv(&L);
            }
            size_t need = strlen(ident) + 2 + strlen(args) + 1;
            char *combined = (char *)triad_arena_alloc(arena, need);
            snprintf(combined, need, "%s(%s)", ident, args);
            TriadLegacyToken t = {0};
            t.kind = TRIAD_LTOK_ANNOT;
            t.text = combined;
            t.line = sl; t.col = sc;
            if (ll_push(&L, t) != 0) goto fail;
            continue;
        }

        /* number — including a leading "-" if the next char is a digit */
        if (isdigit((unsigned char)c) ||
            (c == '-' && L.i + 1 < L.n && isdigit((unsigned char)L.src[L.i + 1]))) {
            int sl = L.line, sc = L.col;
            int is_neg = 0;
            if (c == '-') { ll_adv(&L); is_neg = 1; }

            if (L.i + 1 < L.n && L.src[L.i] == '0' &&
                (L.src[L.i + 1] == 'x' || L.src[L.i + 1] == 'X' ||
                 L.src[L.i + 1] == 'b' || L.src[L.i + 1] == 'B')) {
                char base_char = (char)tolower((unsigned char)L.src[L.i + 1]);
                ll_adv(&L); ll_adv(&L);
                size_t start = L.i;
                if (base_char == 'b') {
                    while (L.i < L.n && (L.src[L.i] == '0' || L.src[L.i] == '1')) ll_adv(&L);
                } else {
                    while (L.i < L.n && isxdigit((unsigned char)L.src[L.i])) ll_adv(&L);
                }
                char *digits = triad_arena_strndup(arena, L.src + start, L.i - start);
                long long val = strtoll(digits, NULL, base_char == 'b' ? 2 : 16);
                if (is_neg) val = -val;
                char buf[32];
                snprintf(buf, sizeof(buf), "%lld", val);
                TriadLegacyToken t = {0};
                t.kind = TRIAD_LTOK_NUMBER;
                t.text = triad_arena_strdup(arena, buf);
                t.line = sl; t.col = sc;
                if (ll_push(&L, t) != 0) goto fail;
                continue;
            }

            /* decimal / float */
            size_t start = is_neg ? L.i - 1 : L.i;
            while (L.i < L.n && isdigit((unsigned char)L.src[L.i])) ll_adv(&L);
            if (L.i < L.n && L.src[L.i] == '.') {
                ll_adv(&L);
                while (L.i < L.n && isdigit((unsigned char)L.src[L.i])) ll_adv(&L);
            }
            if (L.i < L.n && (L.src[L.i] == 'e' || L.src[L.i] == 'E')) {
                ll_adv(&L);
                if (L.i < L.n && (L.src[L.i] == '+' || L.src[L.i] == '-')) ll_adv(&L);
                while (L.i < L.n && isdigit((unsigned char)L.src[L.i])) ll_adv(&L);
            }
            TriadLegacyToken t = {0};
            t.kind = TRIAD_LTOK_NUMBER;
            t.text = triad_arena_strndup(arena, L.src + start, L.i - start);
            t.line = sl; t.col = sc;
            if (ll_push(&L, t) != 0) goto fail;
            continue;
        }

        /* identifier / keyword / opcode */
        if (isalpha((unsigned char)c) || c == '_') {
            int sl = L.line, sc = L.col;
            size_t start = L.i;
            while (L.i < L.n && (isalnum((unsigned char)L.src[L.i]) || L.src[L.i] == '_')) ll_adv(&L);
            size_t len = L.i - start;
            const char *val = triad_arena_strndup(arena, L.src + start, len);
            TriadLegacyTokKind kind;
            if (in_set(L_OPCODES, L.src + start, len))                     kind = TRIAD_LTOK_OPCODE;
            else if (in_set(L_KEYWORDS, L.src + start, len) ||
                     in_set(L_LEGACY_KW, L.src + start, len))              kind = TRIAD_LTOK_KEYWORD;
            else                                                            kind = TRIAD_LTOK_IDENT;
            TriadLegacyToken t = {0};
            t.kind = kind; t.text = val; t.line = sl; t.col = sc;
            if (ll_push(&L, t) != 0) goto fail;
            continue;
        }

        /* string */
        if (c == '"') {
            int sl = L.line, sc = L.col;
            ll_adv(&L);
            size_t start = L.i;
            while (L.i < L.n && L.src[L.i] != '"') {
                if (L.src[L.i] == '\n') { L.line++; L.col = 1; }
                ll_adv(&L);
            }
            if (L.i >= L.n) { llerr(&L, sl, sc, "unterminated string"); goto fail; }
            const char *val = triad_arena_strndup(arena, L.src + start, L.i - start);
            ll_adv(&L);
            TriadLegacyToken t = {0};
            t.kind = TRIAD_LTOK_STRING; t.text = val; t.line = sl; t.col = sc;
            if (ll_push(&L, t) != 0) goto fail;
            continue;
        }

        /* multi-char symbols */
        if (L.i + 1 < L.n) {
            const char *p = L.src + L.i;
            const char *two = NULL;
            if      (p[0] == ':' && p[1] == '=') two = ":=";
            else if (p[0] == '-' && p[1] == '>') two = "->";
            else if (p[0] == '=' && p[1] == '=') two = "==";
            else if (p[0] == '<' && p[1] == '=') two = "<=";
            else if (p[0] == '>' && p[1] == '=') two = ">=";
            if (two) {
                TriadLegacyToken t = {0};
                t.kind = TRIAD_LTOK_SYMBOL;
                t.text = triad_arena_strdup(arena, two);
                t.line = L.line; t.col = L.col;
                if (ll_push(&L, t) != 0) goto fail;
                ll_adv(&L); ll_adv(&L);
                continue;
            }
        }

        /* single-char symbols */
        if (strchr(";,(){}[]:=<>+-*/", c) != NULL) {
            char one[2] = { c, 0 };
            TriadLegacyToken t = {0};
            t.kind = TRIAD_LTOK_SYMBOL;
            t.text = triad_arena_strdup(arena, one);
            t.line = L.line; t.col = L.col;
            if (ll_push(&L, t) != 0) goto fail;
            ll_adv(&L);
            continue;
        }

        llerr(&L, L.line, L.col, "unexpected character '%c'", c);
        goto fail;
    }

    TriadLegacyToken eof = {0};
    eof.kind = TRIAD_LTOK_EOF;
    eof.text = triad_arena_strdup(arena, "");
    eof.line = L.line; eof.col = L.col;
    if (ll_push(&L, eof) != 0) goto fail;

    TriadLegacyToken *arr = (TriadLegacyToken *)triad_arena_alloc(arena, L.toks_len * sizeof(*arr));
    for (size_t k = 0; k < L.toks_len; ++k) arr[k] = L.toks[k];
    free(L.toks);
    out->items = arr;
    out->len   = L.toks_len;
    return 0;

fail:
    free(L.toks);
    out->items = NULL; out->len = 0;
    return -1;
}
