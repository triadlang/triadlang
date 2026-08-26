#include "triad_lsp.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *KEYWORDS[] = {
    "let", "const", "fn", "return", "if", "elif", "else",
    "for", "while", "break", "continue", "class", "type",
    "import", "from", "as", "match", "with", "and", "or",
    "not", "true", "false", "none", "yield", "async", "await",
    "try", "catch", "finally", "throw", "reg", "observe", "run",
    "couple", "pair", "ring", "entity", "world", "in",
};
#define N_KEYWORDS (sizeof(KEYWORDS) / sizeof(KEYWORDS[0]))

static const char *BUILTINS[] = {
    "len", "range", "str", "int", "float", "abs", "min", "max",
    "sorted", "print", "type", "list", "dict", "enumerate",
};
#define N_BUILTINS (sizeof(BUILTINS) / sizeof(BUILTINS[0]))

static const char *TRIAD_KEYWORDS[] = {
    "reg", "observe", "run", "couple", "pair", "ring", "entity", "world",
};
#define N_TRIAD_KEYWORDS (sizeof(TRIAD_KEYWORDS) / sizeof(TRIAD_KEYWORDS[0]))

typedef struct {
    const char        *name;
    const char *const *fns;
    size_t             n;
} StdlibMod;

static const char *MOD_math[] = {
    "sqrt","sin","cos","tan","log","log10","exp","floor","ceil","abs",
    "pi","e","min","max","clamp","pow",
};
static const char *MOD_random[]   = {"random","randint","choice","seed","shuffle","uniform"};
static const char *MOD_io[]       = {"print","input"};
static const char *MOD_string[]   = {"split","join","replace","lower","upper","strip","starts_with","ends_with","contains"};
static const char *MOD_json[]     = {"parse","stringify"};
static const char *MOD_fs[]       = {"read_text","write_text","exists","listdir"};
static const char *MOD_time[]     = {"now","sleep"};
static const char *MOD_collections[] = {"len","range","enumerate","sorted","reversed","zip","map","filter"};

static const StdlibMod STDLIB_MODULES[] = {
    {"math",        MOD_math,        sizeof(MOD_math)        / sizeof(*MOD_math)},
    {"random",      MOD_random,      sizeof(MOD_random)      / sizeof(*MOD_random)},
    {"io",          MOD_io,          sizeof(MOD_io)          / sizeof(*MOD_io)},
    {"string",      MOD_string,      sizeof(MOD_string)      / sizeof(*MOD_string)},
    {"json",        MOD_json,        sizeof(MOD_json)        / sizeof(*MOD_json)},
    {"fs",          MOD_fs,          sizeof(MOD_fs)          / sizeof(*MOD_fs)},
    {"time",        MOD_time,        sizeof(MOD_time)        / sizeof(*MOD_time)},
    {"collections", MOD_collections, sizeof(MOD_collections) / sizeof(*MOD_collections)},
};
#define N_MODS (sizeof(STDLIB_MODULES) / sizeof(*STDLIB_MODULES))

typedef struct { char *data; size_t len, cap; } Buf;
static void buf_init(Buf *b) { b->data=NULL; b->len=b->cap=0; }
static void buf_reserve(Buf *b, size_t n) {
    if (b->len + n + 1 > b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 128;
        while (nc < b->len + n + 1) nc *= 2;
        b->data = (char *)realloc(b->data, nc);
        b->cap = nc;
    }
}
static void buf_append(Buf *b, const char *s, size_t n) {
    buf_reserve(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}
static void buf_puts(Buf *b, const char *s) { buf_append(b, s, strlen(s)); }
static void buf_putc(Buf *b, char c) { buf_reserve(b, 1); b->data[b->len++]=c; b->data[b->len]='\0'; }
static void buf_printf(Buf *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt); va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) { va_end(ap2); return; }
    buf_reserve(b, (size_t)n);
    vsnprintf(b->data + b->len, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}
static void buf_free(Buf *b) { free(b->data); b->data=NULL; b->len=b->cap=0; }
static const char *buf_to_arena(TriadArena *a, Buf *b) {
    char *o = (char *)triad_arena_alloc(a, b->len + 1);
    memcpy(o, b->data, b->len); o[b->len]='\0';
    buf_free(b);
    return o;
}

static void json_string_into(Buf *b, const char *s) {
    buf_putc(b, '"');
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        unsigned char c = *p;
        if      (c == '\\') buf_puts(b, "\\\\");
        else if (c == '"')  buf_puts(b, "\\\"");
        else if (c == '\n') buf_puts(b, "\\n");
        else if (c == '\r') buf_puts(b, "\\r");
        else if (c == '\t') buf_puts(b, "\\t");
        else if (c == '\b') buf_puts(b, "\\b");
        else if (c == '\f') buf_puts(b, "\\f");
        else if (c < 0x20)  buf_printf(b, "\\u%04x", c);
        else                buf_putc(b, (char)c);
    }
    buf_putc(b, '"');
}

typedef struct {
    const char **lines;
    int         *lens;
    size_t       n;
} TextLines;

static TextLines split_text(TriadArena *a, const char *text) {
    TextLines T = {0};
    size_t cap = 32, n = 0;
    const char **arr = (const char **)malloc(sizeof(char *) * cap);
    int          *ls = (int *)malloc(sizeof(int) * cap);
    const char *p = text;
    while (1) {
        const char *start = p;
        while (*p && *p != '\n') p++;
        size_t L = (size_t)(p - start);
        char *line = (char *)triad_arena_alloc(a, L + 1);
        memcpy(line, start, L); line[L] = '\0';
        if (n + 1 > cap) {
            cap *= 2;
            arr = (const char **)realloc(arr, sizeof(char *) * cap);
            ls  = (int *)realloc(ls, sizeof(int) * cap);
        }
        arr[n] = line; ls[n] = (int)L;
        n++;
        if (!*p) break;
        p++;
    }
    T.n = n;
    T.lines = (const char **)triad_arena_alloc(a, sizeof(char *) * n);
    T.lens  = (int *)triad_arena_alloc(a, sizeof(int) * n);
    memcpy((void *)T.lines, arr, sizeof(char *) * n);
    memcpy(T.lens, ls, sizeof(int) * n);
    free(arr); free(ls);
    return T;
}

static int is_ident(char c) { return isalnum((unsigned char)c) || c == '_'; }

static int lstrip_n(const char *s) {
    int n = 0;
    while (s[n] == ' ' || s[n] == '\t' || s[n] == '\v' || s[n] == '\f' || s[n] == '\r') n++;
    return n;
}

static size_t match_decl(const char *stripped, const char *kw,
                         const char **out_name) {
    size_t kl = strlen(kw);
    if (strncmp(stripped, kw, kl) != 0) return 0;
    const char *p = stripped + kl;
    if (!isspace((unsigned char)*p)) return 0;
    while (isspace((unsigned char)*p)) p++;
    const char *s = p;
    while (is_ident(*p)) p++;
    if (p == s) return 0;
    *out_name = s;
    return (size_t)(p - s);
}

static size_t match_let_const(const char *stripped, const char **out_name) {
    size_t L;
    if ((L = match_decl(stripped, "let", out_name))) return L;
    if ((L = match_decl(stripped, "const", out_name))) return L;
    return 0;
}

void triad_lsp_parse_symbols(TriadArena *a, const char *text,
                             TriadLspSymbol **out, size_t *out_len) {
    TextLines T = split_text(a, text);
    size_t cap = 16, n = 0;
    TriadLspSymbol *arr = (TriadLspSymbol *)malloc(sizeof(*arr) * cap);

    for (size_t i = 0; i < T.n; ++i) {
        const char *line = T.lines[i];
        int indent = lstrip_n(line);
        const char *str = line + indent;
        const char *name; size_t L;

        if ((L = match_let_const(str, &name))) {
            if (n + 1 > cap) { cap *= 2; arr = (TriadLspSymbol *)realloc(arr, sizeof(*arr) * cap); }
            char *nm = (char *)triad_arena_alloc(a, L + 1);
            memcpy(nm, name, L); nm[L] = '\0';
            arr[n++] = (TriadLspSymbol){nm, "variable", (int)i, indent};
            continue;
        }
        if ((L = match_decl(str, "fn", &name))) {
            if (n + 1 > cap) { cap *= 2; arr = (TriadLspSymbol *)realloc(arr, sizeof(*arr) * cap); }
            char *nm = (char *)triad_arena_alloc(a, L + 1);
            memcpy(nm, name, L); nm[L] = '\0';
            arr[n++] = (TriadLspSymbol){nm, "function", (int)i, indent};
            continue;
        }
        if ((L = match_decl(str, "class", &name))) {
            if (n + 1 > cap) { cap *= 2; arr = (TriadLspSymbol *)realloc(arr, sizeof(*arr) * cap); }
            char *nm = (char *)triad_arena_alloc(a, L + 1);
            memcpy(nm, name, L); nm[L] = '\0';
            arr[n++] = (TriadLspSymbol){nm, "class", (int)i, indent};
            continue;
        }
        if ((L = match_decl(str, "type", &name))) {
            if (n + 1 > cap) { cap *= 2; arr = (TriadLspSymbol *)realloc(arr, sizeof(*arr) * cap); }
            char *nm = (char *)triad_arena_alloc(a, L + 1);
            memcpy(nm, name, L); nm[L] = '\0';
            arr[n++] = (TriadLspSymbol){nm, "class", (int)i, indent};
            continue;
        }
    }
    *out_len = n;
    if (n == 0) { *out = NULL; free(arr); return; }
    TriadLspSymbol *final = (TriadLspSymbol *)triad_arena_alloc(a, sizeof(*final) * n);
    memcpy(final, arr, sizeof(*final) * n);
    free(arr);
    *out = final;
}

const char *triad_lsp_word_at(TriadArena *a, const char *line, int char_pos) {
    int L = (int)strlen(line);
    int c = char_pos;
    if (c >= L) c = L - 1;
    if (c < 0) return NULL;
    if (!is_ident(line[c])) {
        if (c > 0 && is_ident(line[c-1])) c--;
        else return NULL;
    }
    int start = c;
    while (start > 0 && is_ident(line[start-1])) start--;
    int end = c + 1;
    while (end < L && is_ident(line[end])) end++;
    char *out = (char *)triad_arena_alloc(a, (size_t)(end - start) + 1);
    memcpy(out, line + start, (size_t)(end - start));
    out[end - start] = '\0';
    return out;
}

static void emit_completion_item(Buf *b, const char *label, int kind, const char *detail) {
    buf_puts(b, "{\"label\": ");
    json_string_into(b, label);
    buf_printf(b, ", \"kind\": %d, \"detail\": ", kind);
    json_string_into(b, detail);
    buf_putc(b, '}');
}

static int set_has(const char **set, size_t n, const char *s) {
    for (size_t i = 0; i < n; ++i) if (strcmp(set[i], s) == 0) return 1;
    return 0;
}

const char *triad_lsp_completions_json(TriadArena *a, const char *text,
                                       int line_num, int char_pos) {
    TextLines T = split_text(a, text);
    Buf out; buf_init(&out);

    if (line_num >= (int)T.n) {
        buf_puts(&out, "{\"isIncomplete\": false, \"items\": []}");
        return buf_to_arena(a, &out);
    }
    const char *cur = T.lines[line_num];
    int curL = T.lens[line_num];
    int char_clamp = char_pos > curL ? curL : char_pos;
    if (char_clamp < 0) char_clamp = 0;

    char *prefix = (char *)triad_arena_alloc(a, (size_t)char_clamp + 1);
    memcpy(prefix, cur, (size_t)char_clamp);
    prefix[char_clamp] = '\0';

    int plen = (int)strlen(prefix);
    int j = plen;
    while (j > 0 && is_ident(prefix[j-1])) j--;
    int partial_start = j;
    if (j > 0 && prefix[j-1] == '.') {
        int dot_pos = j - 1;
        int k = dot_pos;
        while (k > 0 && is_ident(prefix[k-1])) k--;
        if (k < dot_pos) {

            char *modname = (char *)triad_arena_alloc(a, (size_t)(dot_pos - k) + 1);
            memcpy(modname, prefix + k, (size_t)(dot_pos - k));
            modname[dot_pos - k] = '\0';
            const char *partial = prefix + partial_start;
            for (size_t i = 0; i < N_MODS; ++i) {
                if (strcmp(STDLIB_MODULES[i].name, modname) == 0) {
                    Buf items; buf_init(&items);
                    int first = 1;
                    for (size_t f = 0; f < STDLIB_MODULES[i].n; ++f) {
                        const char *fn = STDLIB_MODULES[i].fns[f];
                        if (strncmp(fn, partial, strlen(partial)) == 0) {
                            if (!first) buf_puts(&items, ", ");
                            char detail[128];
                            snprintf(detail, sizeof detail, "%s.%s", modname, fn);
                            emit_completion_item(&items, fn, 3, detail);
                            first = 0;
                        }
                    }
                    buf_puts(&out, "{\"isIncomplete\": false, \"items\": [");
                    if (items.len) buf_append(&out, items.data, items.len);
                    buf_puts(&out, "]}");
                    buf_free(&items);
                    return buf_to_arena(a, &out);
                }
            }
        }
    }

    const char *partial = prefix + partial_start;
    int partial_len = plen - partial_start;
    int prefix_ends_space = (plen > 0 && prefix[plen-1] == ' ');
    if (partial_len == 0 && !prefix_ends_space) {
        buf_puts(&out, "{\"isIncomplete\": false, \"items\": []}");
        return buf_to_arena(a, &out);
    }

    TriadLspSymbol *syms = NULL; size_t nsyms = 0;
    triad_lsp_parse_symbols(a, text, &syms, &nsyms);

    Buf items; buf_init(&items);
    int first = 1;

    const char **locals = (const char **)triad_arena_alloc(a, sizeof(char *) * (nsyms + 1));
    size_t nlocals = 0;
    for (size_t i = 0; i < nsyms; ++i) {
        if (strncmp(syms[i].name, partial, (size_t)partial_len) == 0) {
            int kind;
            if (strcmp(syms[i].kind, "variable") == 0) kind = 13;
            else if (strcmp(syms[i].kind, "function") == 0) kind = 12;
            else kind = 7;
            if (!first) buf_puts(&items, ", ");
            emit_completion_item(&items, syms[i].name, kind, syms[i].kind);
            first = 0;
        }
        locals[nlocals++] = syms[i].name;
    }

    int prefix_endswith_dot = (plen > 0 && prefix[plen-1] == '.');
    if (!prefix_endswith_dot) {

        for (size_t i = 0; i < N_KEYWORDS; ++i) {
            const char *kw = KEYWORDS[i];
            if (strncmp(kw, partial, (size_t)partial_len) == 0 &&
                !set_has(locals, nlocals, kw)) {
                if (!first) buf_puts(&items, ", ");
                emit_completion_item(&items, kw, 14, "keyword");
                first = 0;
            }
        }
        for (size_t i = 0; i < N_BUILTINS; ++i) {
            const char *kw = BUILTINS[i];
            if (strncmp(kw, partial, (size_t)partial_len) == 0 &&
                !set_has(locals, nlocals, kw)) {
                if (!first) buf_puts(&items, ", ");
                emit_completion_item(&items, kw, 3, "builtin");
                first = 0;
            }
        }
        for (size_t i = 0; i < N_MODS; ++i) {
            const char *kw = STDLIB_MODULES[i].name;
            if (strncmp(kw, partial, (size_t)partial_len) == 0) {
                if (!first) buf_puts(&items, ", ");
                emit_completion_item(&items, kw, 9, "module");
                first = 0;
            }
        }
        for (size_t i = 0; i < N_TRIAD_KEYWORDS; ++i) {
            const char *kw = TRIAD_KEYWORDS[i];
            if (strncmp(kw, partial, (size_t)partial_len) == 0 &&
                !set_has(locals, nlocals, kw)) {
                if (!first) buf_puts(&items, ", ");
                emit_completion_item(&items, kw, 14, "triad");
                first = 0;
            }
        }
    }

    buf_puts(&out, "{\"isIncomplete\": false, \"items\": [");
    if (items.len) buf_append(&out, items.data, items.len);
    buf_puts(&out, "]}");
    buf_free(&items);
    return buf_to_arena(a, &out);
}

static int matches_def(const char *stripped, const char *kw, const char *word) {
    size_t kl = strlen(kw);
    if (strncmp(stripped, kw, kl) != 0) return 0;
    const char *p = stripped + kl;
    if (!isspace((unsigned char)*p)) return 0;
    while (isspace((unsigned char)*p)) p++;
    size_t wl = strlen(word);
    if (strncmp(p, word, wl) != 0) return 0;
    char nxt = p[wl];
    if (is_ident(nxt)) return 0;
    return 1;
}

const char *triad_lsp_definition_json(TriadArena *a, const char *text,
                                      int line_num, int char_pos,
                                      const char *uri) {
    TextLines T = split_text(a, text);
    if (line_num >= (int)T.n) return NULL;
    const char *word = triad_lsp_word_at(a, T.lines[line_num], char_pos);
    if (!word) return NULL;

    for (size_t i = 0; i < T.n; ++i) {
        const char *line = T.lines[i];
        int indent = lstrip_n(line);
        const char *str = line + indent;
        int strL = T.lens[i] - indent;
        int hit = 0;
        if (matches_def(str, "let", word)   ||
            matches_def(str, "const", word) ||
            matches_def(str, "fn", word)    ||
            matches_def(str, "class", word) ||
            matches_def(str, "type", word)) hit = 1;
        if (hit) {
            Buf b; buf_init(&b);
            buf_puts(&b, "{\"uri\": ");
            json_string_into(&b, uri);
            buf_printf(&b, ", \"range\": {\"start\": {\"line\": %zu, \"character\": %d}, \"end\": {\"line\": %zu, \"character\": %d}}}",
                       i, indent, i, indent + strL);
            return buf_to_arena(a, &b);
        }
    }
    return NULL;
}

const char *triad_lsp_hover_json(TriadArena *a, const char *text,
                                 int line_num, int char_pos) {
    TextLines T = split_text(a, text);
    if (line_num >= (int)T.n) return NULL;
    const char *word = triad_lsp_word_at(a, T.lines[line_num], char_pos);
    if (!word) return NULL;

    for (size_t i = 0; i < N_MODS; ++i) {
        if (strcmp(STDLIB_MODULES[i].name, word) == 0) {
            Buf v; buf_init(&v);
            buf_printf(&v, "**module %s**\n\nExports: `", STDLIB_MODULES[i].name);
            for (size_t f = 0; f < STDLIB_MODULES[i].n; ++f) {
                if (f) buf_puts(&v, ", ");
                buf_puts(&v, STDLIB_MODULES[i].fns[f]);
            }
            buf_puts(&v, "`");
            char *value = (char *)triad_arena_alloc(a, v.len + 1);
            memcpy(value, v.data, v.len + 1);
            buf_free(&v);
            Buf out; buf_init(&out);
            buf_puts(&out, "{\"contents\": {\"kind\": \"markdown\", \"value\": ");
            json_string_into(&out, value);
            buf_puts(&out, "}}");
            return buf_to_arena(a, &out);
        }
    }

    for (size_t i = 0; i < N_MODS; ++i) {
        for (size_t f = 0; f < STDLIB_MODULES[i].n; ++f) {
            if (strcmp(STDLIB_MODULES[i].fns[f], word) == 0) {
                char value[512];
                snprintf(value, sizeof value,
                         "**%s.%s**\n\nStdlib function from `%s`",
                         STDLIB_MODULES[i].name, word, STDLIB_MODULES[i].name);
                Buf out; buf_init(&out);
                buf_puts(&out, "{\"contents\": {\"kind\": \"markdown\", \"value\": ");
                json_string_into(&out, value);
                buf_puts(&out, "}}");
                return buf_to_arena(a, &out);
            }
        }
    }

    TriadLspSymbol *syms = NULL; size_t nsyms = 0;
    triad_lsp_parse_symbols(a, text, &syms, &nsyms);
    for (size_t i = 0; i < nsyms; ++i) {
        if (strcmp(syms[i].name, word) == 0) {
            char value[256];
            snprintf(value, sizeof value,
                     "**%s** (%s)\n\nDefined at line %d",
                     word, syms[i].kind, syms[i].line + 1);
            Buf out; buf_init(&out);
            buf_puts(&out, "{\"contents\": {\"kind\": \"markdown\", \"value\": ");
            json_string_into(&out, value);
            buf_puts(&out, "}}");
            return buf_to_arena(a, &out);
        }
    }

    for (size_t i = 0; i < N_KEYWORDS; ++i) {
        if (strcmp(KEYWORDS[i], word) == 0) {
            char value[128];
            snprintf(value, sizeof value, "**%s** — keyword", word);
            Buf out; buf_init(&out);
            buf_puts(&out, "{\"contents\": {\"kind\": \"markdown\", \"value\": ");
            json_string_into(&out, value);
            buf_puts(&out, "}}");
            return buf_to_arena(a, &out);
        }
    }
    return NULL;
}

const char *triad_lsp_document_symbols_json(TriadArena *a, const char *text,
                                            const char *uri) {
    TriadLspSymbol *syms = NULL; size_t nsyms = 0;
    triad_lsp_parse_symbols(a, text, &syms, &nsyms);
    Buf out; buf_init(&out);
    buf_putc(&out, '[');
    for (size_t i = 0; i < nsyms; ++i) {
        if (i) buf_puts(&out, ", ");
        int kind;
        if      (strcmp(syms[i].kind, "variable") == 0) kind = 6;
        else if (strcmp(syms[i].kind, "function") == 0) kind = 12;
        else if (strcmp(syms[i].kind, "class") == 0)    kind = 5;
        else                                            kind = 6;
        size_t nlen = strlen(syms[i].name);
        buf_puts(&out, "{\"name\": ");
        json_string_into(&out, syms[i].name);
        buf_printf(&out, ", \"kind\": %d, \"location\": {\"uri\": ", kind);
        json_string_into(&out, uri);
        buf_printf(&out,
            ", \"range\": {\"start\": {\"line\": %d, \"character\": %d}, "
            "\"end\": {\"line\": %d, \"character\": %d}}}}",
            syms[i].line, syms[i].col,
            syms[i].line, syms[i].col + (int)nlen);
    }
    buf_putc(&out, ']');
    return buf_to_arena(a, &out);
}

const char *triad_lsp_initialize_result_json(TriadArena *a) {

    static const char *S =
        "{\"capabilities\": {\"completionProvider\": {\"triggerCharacters\": [\".\", \" \"]}, "
        "\"definitionProvider\": true, \"hoverProvider\": true, "
        "\"documentSymbolProvider\": true, "
        "\"textDocumentSync\": {\"openClose\": true, \"change\": 1}}}";
    size_t L = strlen(S);
    char *out = (char *)triad_arena_alloc(a, L + 1);
    memcpy(out, S, L + 1);
    return out;
}
