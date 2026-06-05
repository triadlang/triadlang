/* triad_lexer.c — Port of frontend/lexer_universal.py. */
#include "triad_frontend.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Keyword table (matches KEYWORDS in lexer_universal.py) ─────── */

static const char *const KEYWORDS[] = {
    "let", "const", "fn", "return", "if", "else", "elif",
    "for", "in", "while", "break", "continue",
    "type", "class", "super", "import", "from", "as",
    "true", "false", "none",
    "and", "or", "not",
    "try", "catch", "finally", "throw",
    "self",
    "match", "case",
    "yield", "async", "await",
    "reg", "entity", "world", "couple", "pair", "ring",
    "observe", "OBSERVE", "run",
    "evolve", "sequence", "via", "each_for",
    "substrate", "composed_of", "assert",
    "persistent", "extended", "structurally_open", "non_trivial_memory",
    "atomic", "anti_collapsed",
    "over_seeds",
    NULL
};

static int is_keyword(const char *s, size_t n) {
    for (size_t k = 0; KEYWORDS[k]; ++k) {
        const char *kw = KEYWORDS[k];
        size_t kn = strlen(kw);
        if (kn == n && memcmp(kw, s, n) == 0) return 1;
    }
    return 0;
}

/* ── Lexer state ────────────────────────────────────────────────── */

typedef struct {
    TriadArena   *arena;
    const char   *src;
    size_t        n;
    size_t        i;
    int           line;
    int           col;
    const char   *file;
    TriadDiag    *diag;

    /* growable token buffer */
    TriadToken   *toks;
    size_t        toks_len;
    size_t        toks_cap;
} Lex;

static void set_lex_error(Lex *L, int line, int col, const char *fmt, ...) {
    if (!L->diag) return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    L->diag->line = line;
    L->diag->col  = col;
    L->diag->file = L->file;
    L->diag->kind = "LEX";
    L->diag->msg  = triad_arena_strdup(L->arena, buf);
}

static int push_tok(Lex *L, TriadToken t) {
    if (L->toks_len == L->toks_cap) {
        size_t nc = L->toks_cap ? L->toks_cap * 2 : 64;
        TriadToken *nt = (TriadToken *)realloc(L->toks, nc * sizeof(TriadToken));
        if (!nt) return -1;
        L->toks = nt;
        L->toks_cap = nc;
    }
    L->toks[L->toks_len++] = t;
    return 0;
}

static char peek(Lex *L, size_t off) {
    size_t p = L->i + off;
    return (p < L->n) ? L->src[p] : '\0';
}

static char adv(Lex *L) {
    char c = L->src[L->i++];
    L->col++;
    return c;
}

static int is_ident_start(char c) { return isalpha((unsigned char)c) || c == '_'; }
static int is_ident_cont(char c)  { return isalnum((unsigned char)c) || c == '_'; }

/* growable byte buffer using arena snapshots */
typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} Buf;

static int buf_push(Buf *b, char c) {
    if (b->len + 1 >= b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 32;
        char *nd = (char *)realloc(b->data, nc);
        if (!nd) return -1;
        b->data = nd;
        b->cap  = nc;
    }
    b->data[b->len++] = c;
    return 0;
}

static const char *buf_dup(TriadArena *a, Buf *b) {
    if (!b->data) return triad_arena_strdup(a, "");
    char *out = triad_arena_strndup(a, b->data, b->len);
    return out;
}

static void buf_free(Buf *b) { free(b->data); b->data = NULL; b->len = b->cap = 0; }

/* ── Lex sub-routines ───────────────────────────────────────────── */

/* Returns 0 on success, -1 on error (diag set). */
static int lex_block_comment(Lex *L) {
    int sl = L->line, sc = L->col;
    adv(L); adv(L); /* consume "/" "*" */
    while (L->i < L->n) {
        if (L->src[L->i] == '*' && peek(L, 1) == '/') {
            adv(L); adv(L);
            return 0;
        }
        if (L->src[L->i] == '\n') {
            adv(L);
            L->line++;
            L->col = 1;
        } else {
            adv(L);
        }
    }
    set_lex_error(L, sl, sc, "unterminated block comment");
    return -1;
}

static int lex_string(Lex *L) {
    int sl = L->line, sc = L->col;
    adv(L); /* opening " */
    Buf buf = {0};
    while (L->i < L->n && L->src[L->i] != '"') {
        if (L->src[L->i] == '\\') {
            adv(L);
            if (L->i < L->n) {
                char esc = adv(L);
                char out = esc;
                switch (esc) {
                    case 'n': out = '\n'; break;
                    case 't': out = '\t'; break;
                    case '\\': out = '\\'; break;
                    case '"': out = '"'; break;
                    default: out = esc; break;
                }
                if (buf_push(&buf, out) != 0) { buf_free(&buf); set_lex_error(L, sl, sc, "oom in string"); return -1; }
            }
            continue;
        }
        if (L->src[L->i] == '\n') {
            L->line++;
            L->col = 1;
        }
        if (buf_push(&buf, L->src[L->i]) != 0) { buf_free(&buf); set_lex_error(L, sl, sc, "oom in string"); return -1; }
        adv(L);
    }
    if (L->i >= L->n) {
        buf_free(&buf);
        set_lex_error(L, sl, sc, "unterminated string");
        return -1;
    }
    adv(L); /* closing " */
    TriadToken t = {0};
    t.kind = TRIAD_TOK_STRING;
    t.text = buf_dup(L->arena, &buf);
    t.line = sl;
    t.col  = sc;
    buf_free(&buf);
    return push_tok(L, t);
}

static int lex_fstring(Lex *L) {
    int sl = L->line, sc = L->col;
    adv(L); /* f */
    adv(L); /* opening " */

    /* Collect parts: alternating ("str", text) / ("expr", source). */
    TriadFStringPart *parts = NULL;
    size_t parts_len = 0, parts_cap = 0;
    Buf buf = {0};

    #define FLUSH_LITERAL() do { \
        if (buf.len > 0) { \
            if (parts_len == parts_cap) { \
                parts_cap = parts_cap ? parts_cap * 2 : 4; \
                parts = (TriadFStringPart *)realloc(parts, parts_cap * sizeof(TriadFStringPart)); \
            } \
            TriadFStringPart pp; pp.is_expr = 0; pp.text = buf_dup(L->arena, &buf); \
            parts[parts_len++] = pp; \
            buf.len = 0; \
        } \
    } while (0)

    while (L->i < L->n && L->src[L->i] != '"') {
        if (L->src[L->i] == '{') {
            FLUSH_LITERAL();
            adv(L); /* { */
            Buf ebuf = {0};
            int depth = 1;
            while (L->i < L->n && depth > 0) {
                if (L->src[L->i] == '{') depth++;
                else if (L->src[L->i] == '}') depth--;
                if (depth > 0) {
                    if (L->src[L->i] == '\n') { L->line++; L->col = 1; }
                    buf_push(&ebuf, L->src[L->i]);
                    adv(L);
                } else {
                    adv(L); /* closing } */
                }
            }
            if (parts_len == parts_cap) {
                parts_cap = parts_cap ? parts_cap * 2 : 4;
                parts = (TriadFStringPart *)realloc(parts, parts_cap * sizeof(TriadFStringPart));
            }
            TriadFStringPart pp;
            pp.is_expr = 1;
            pp.text    = buf_dup(L->arena, &ebuf);
            parts[parts_len++] = pp;
            buf_free(&ebuf);
            continue;
        }
        if (L->src[L->i] == '\\') {
            adv(L);
            if (L->i < L->n) {
                char esc = adv(L);
                char out = esc;
                switch (esc) {
                    case 'n': out = '\n'; break;
                    case 't': out = '\t'; break;
                    case '\\': out = '\\'; break;
                    case '"': out = '"'; break;
                    default: out = esc; break;
                }
                buf_push(&buf, out);
            }
            continue;
        }
        if (L->src[L->i] == '\n') { L->line++; L->col = 1; }
        buf_push(&buf, L->src[L->i]);
        adv(L);
    }
    if (L->i >= L->n) {
        buf_free(&buf);
        free(parts);
        set_lex_error(L, sl, sc, "unterminated f-string");
        return -1;
    }
    adv(L); /* closing " */
    FLUSH_LITERAL();
    buf_free(&buf);
    #undef FLUSH_LITERAL

    /* Move parts into the arena. */
    TriadFStringPart *aparts = (TriadFStringPart *)triad_arena_alloc(
        L->arena, parts_len * sizeof(TriadFStringPart));
    for (size_t k = 0; k < parts_len; ++k) aparts[k] = parts[k];
    free(parts);

    TriadToken t = {0};
    t.kind      = TRIAD_TOK_FSTRING;
    t.text      = NULL;
    t.parts     = aparts;
    t.parts_len = parts_len;
    t.line      = sl;
    t.col       = sc;
    return push_tok(L, t);
}

static int lex_number(Lex *L) {
    int sl = L->line, sc = L->col;
    size_t start = L->i;
    char c = L->src[L->i];
    char p1 = peek(L, 1);

    if (c == '0' && (p1 == 'x' || p1 == 'X' || p1 == 'b' || p1 == 'B')) {
        adv(L); adv(L);
        while (L->i < L->n && (isalnum((unsigned char)L->src[L->i]) || L->src[L->i] == '_')) adv(L);
    } else {
        while (L->i < L->n && isdigit((unsigned char)L->src[L->i])) adv(L);
        if (L->i < L->n && L->src[L->i] == '.' && peek(L, 1) != '.') {
            adv(L);
            while (L->i < L->n && isdigit((unsigned char)L->src[L->i])) adv(L);
        }
        if (L->i < L->n && (L->src[L->i] == 'e' || L->src[L->i] == 'E')) {
            adv(L);
            if (L->i < L->n && (L->src[L->i] == '+' || L->src[L->i] == '-')) adv(L);
            while (L->i < L->n && isdigit((unsigned char)L->src[L->i])) adv(L);
        }
    }

    TriadToken t = {0};
    t.kind = TRIAD_TOK_NUMBER;
    t.text = triad_arena_strndup(L->arena, L->src + start, L->i - start);
    t.line = sl;
    t.col  = sc;
    return push_tok(L, t);
}

static int try_lex_negative_number(Lex *L) {
    /* Mirrors the Python "negative number after operator context" rule.
     * The rule: if previous token is None, or (SYMBOL/KEYWORD and value not in (")", "]")).
     * Note Python expression has subtle precedence; we reproduce it as-is. */
    if (L->src[L->i] != '-' || !isdigit((unsigned char)peek(L, 1))) return 0;
    int eligible;
    if (L->toks_len == 0) {
        eligible = 1;
    } else {
        TriadToken *prev = &L->toks[L->toks_len - 1];
        int is_sym_or_kw = (prev->kind == TRIAD_TOK_SYMBOL || prev->kind == TRIAD_TOK_KEYWORD);
        const char *v = prev->text ? prev->text : "";
        int is_close = (strcmp(v, ")") == 0 || strcmp(v, "]") == 0);
        /* Python: `prev is None or prev.kind in ("SYMBOL","KEYWORD") and prev.value not in (")", "]")`
         * `and` binds tighter than `or`, so the parenthesisation is:
         *   prev is None or (is_sym_or_kw and not is_close)
         */
        eligible = (is_sym_or_kw && !is_close);
    }
    if (!eligible) return 0;

    int sl = L->line, sc = L->col;
    size_t start = L->i;
    adv(L); /* - */
    while (L->i < L->n && isdigit((unsigned char)L->src[L->i])) adv(L);
    if (L->i < L->n && L->src[L->i] == '.') {
        adv(L);
        while (L->i < L->n && isdigit((unsigned char)L->src[L->i])) adv(L);
    }
    TriadToken t = {0};
    t.kind = TRIAD_TOK_NUMBER;
    t.text = triad_arena_strndup(L->arena, L->src + start, L->i - start);
    t.line = sl;
    t.col  = sc;
    return (push_tok(L, t) == 0) ? 1 : -1;
}

static int lex_ident_or_kw(Lex *L) {
    int sl = L->line, sc = L->col;
    size_t start = L->i;
    while (L->i < L->n && is_ident_cont(L->src[L->i])) adv(L);
    size_t len = L->i - start;
    const char *val = triad_arena_strndup(L->arena, L->src + start, len);
    TriadToken t = {0};
    t.kind = is_keyword(L->src + start, len) ? TRIAD_TOK_KEYWORD : TRIAD_TOK_IDENT;
    t.text = val;
    t.line = sl;
    t.col  = sc;
    return push_tok(L, t);
}

static int is_multi_sym(const char *p) {
    static const char *const TWO[] = {
        "==", "!=", "<=", ">=", "->", "**", "+=", "-=", "*=", "/=", "=>", NULL
    };
    for (size_t k = 0; TWO[k]; ++k) {
        if (p[0] == TWO[k][0] && p[1] == TWO[k][1]) return 1;
    }
    return 0;
}

static int is_single_sym(char c) {
    return strchr("+-*/%=<>(){}[];:,.!@&|^~", c) != NULL;
}

/* ── Entry ──────────────────────────────────────────────────────── */

int triad_tokenize(TriadArena *arena, const char *src, const char *file,
                   TriadTokenList *out, TriadDiag *diag)
{
    Lex L = {0};
    L.arena = arena;
    L.src   = src ? src : "";
    L.n     = strlen(L.src);
    L.line  = 1;
    L.col   = 1;
    L.file  = file ? triad_arena_strdup(arena, file) : "";
    L.diag  = diag;

    if (diag) { diag->line = 0; diag->col = 0; diag->file = L.file; diag->kind = NULL; diag->msg = NULL; }

    while (L.i < L.n) {
        char c = L.src[L.i];

        if (c == '\n') { adv(&L); L.line++; L.col = 1; continue; }
        if (c == ' ' || c == '\t' || c == '\r') { adv(&L); continue; }

        if (c == '/' && peek(&L, 1) == '/') {
            while (L.i < L.n && L.src[L.i] != '\n') adv(&L);
            continue;
        }
        if (c == '/' && peek(&L, 1) == '*') {
            if (lex_block_comment(&L) != 0) goto fail;
            continue;
        }
        if (c == '#') {
            while (L.i < L.n && L.src[L.i] != '\n') adv(&L);
            continue;
        }

        if (c == 'f' && peek(&L, 1) == '"') {
            if (lex_fstring(&L) != 0) goto fail;
            continue;
        }

        if (c == '"') {
            if (lex_string(&L) != 0) goto fail;
            continue;
        }

        if (isdigit((unsigned char)c) || (c == '.' && isdigit((unsigned char)peek(&L, 1)))) {
            if (lex_number(&L) != 0) goto fail;
            continue;
        }

        if (c == '-' && isdigit((unsigned char)peek(&L, 1))) {
            int r = try_lex_negative_number(&L);
            if (r < 0) goto fail;
            if (r == 1) continue;
            /* else fall through to symbol handling */
        }

        if (is_ident_start(c)) {
            if (lex_ident_or_kw(&L) != 0) goto fail;
            continue;
        }

        /* multi-char symbols */
        if (L.i + 1 < L.n && is_multi_sym(L.src + L.i)) {
            int sl = L.line, sc = L.col;
            char two[3] = { L.src[L.i], L.src[L.i + 1], 0 };
            TriadToken t = {0};
            t.kind = TRIAD_TOK_SYMBOL;
            t.text = triad_arena_strdup(arena, two);
            t.line = sl;
            t.col  = sc;
            if (push_tok(&L, t) != 0) goto fail;
            adv(&L); adv(&L);
            continue;
        }

        if (is_single_sym(c)) {
            int sl = L.line, sc = L.col;
            char one[2] = { c, 0 };
            TriadToken t = {0};
            t.kind = TRIAD_TOK_SYMBOL;
            t.text = triad_arena_strdup(arena, one);
            t.line = sl;
            t.col  = sc;
            if (push_tok(&L, t) != 0) goto fail;
            adv(&L);
            continue;
        }

        set_lex_error(&L, L.line, L.col, "unexpected character '%c'", c);
        goto fail;
    }

    /* EOF token */
    TriadToken eof = {0};
    eof.kind = TRIAD_TOK_EOF;
    eof.text = triad_arena_strdup(arena, "");
    eof.line = L.line;
    eof.col  = L.col;
    if (push_tok(&L, eof) != 0) goto fail;

    /* Move into arena-owned array. */
    TriadToken *aitems = (TriadToken *)triad_arena_alloc(arena, L.toks_len * sizeof(TriadToken));
    for (size_t k = 0; k < L.toks_len; ++k) aitems[k] = L.toks[k];
    free(L.toks);
    out->items = aitems;
    out->len   = L.toks_len;
    return 0;

fail:
    free(L.toks);
    out->items = NULL;
    out->len   = 0;
    return -1;
}
