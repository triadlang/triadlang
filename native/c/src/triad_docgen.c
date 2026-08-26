#include "triad_docgen.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char  *data;
    size_t len, cap;
} Buf;

static void buf_init(Buf *b) { b->data=NULL; b->len=b->cap=0; }
static void buf_reserve(Buf *b, size_t need) {
    if (b->len + need + 1 > b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 128;
        while (nc < b->len + need + 1) nc *= 2;
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

typedef struct {
    const char **lines;
    size_t       n;
} Lines;

static Lines split_lines(TriadArena *a, const char *src) {
    Lines L = {0};
    size_t cap = 64, n = 0;
    const char **arr = (const char **)malloc(sizeof(char *) * cap);
    const char *p = src;
    while (1) {
        const char *start = p;
        while (*p && *p != '\n') p++;
        size_t len = (size_t)(p - start);
        char *line = (char *)triad_arena_alloc(a, len + 1);
        memcpy(line, start, len);
        line[len] = '\0';
        if (n + 1 > cap) { cap *= 2; arr = (const char **)realloc(arr, sizeof(char *) * cap); }
        arr[n++] = line;
        if (!*p) break;
        p++;
    }
    L.n = n;
    L.lines = (const char **)triad_arena_alloc(a, sizeof(char *) * n);
    memcpy((void *)L.lines, arr, sizeof(char *) * n);
    free(arr);
    return L;
}

static int leading_spaces(const char *s) {
    int n = 0;
    while (s[n] == ' ' || s[n] == '\t') n++;
    return n;
}
static const char *lstrip(const char *s) { return s + leading_spaces(s); }
static int starts_with(const char *s, const char *p) {
    return strncmp(s, p, strlen(p)) == 0;
}

static const char *strip_dup(TriadArena *a, const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    size_t L = strlen(s);
    while (L > 0 && (s[L-1] == ' ' || s[L-1] == '\t' || s[L-1] == '\r')) L--;
    char *out = (char *)triad_arena_alloc(a, L + 1);
    memcpy(out, s, L); out[L] = '\0';
    return out;
}

static int is_ident_char(char c) { return isalnum((unsigned char)c) || c == '_'; }

static int match_fn(const char *line, const char **out_name, size_t *out_name_len,
                    const char **out_params, size_t *out_params_len) {
    if (!starts_with(line, "fn")) return 0;
    const char *p = line + 2;
    if (!isspace((unsigned char)*p)) return 0;
    while (isspace((unsigned char)*p)) p++;
    const char *ns = p;
    while (is_ident_char(*p)) p++;
    if (p == ns) return 0;
    *out_name = ns;
    *out_name_len = (size_t)(p - ns);
    while (isspace((unsigned char)*p)) p++;
    if (*p != '(') return 0;
    p++;
    const char *ps = p;
    while (*p && *p != ')') p++;
    if (*p != ')') return 0;
    *out_params = ps;
    *out_params_len = (size_t)(p - ps);
    return 1;
}

static int match_class(const char *line, const char **out_name, size_t *out_name_len,
                       const char **out_parent, size_t *out_parent_len) {
    if (!starts_with(line, "class")) return 0;
    const char *p = line + 5;
    if (!isspace((unsigned char)*p)) return 0;
    while (isspace((unsigned char)*p)) p++;
    const char *ns = p;
    while (is_ident_char(*p)) p++;
    if (p == ns) return 0;
    *out_name = ns;
    *out_name_len = (size_t)(p - ns);
    *out_parent = NULL;
    *out_parent_len = 0;
    while (isspace((unsigned char)*p)) p++;
    if (*p == '<') {
        p++;
        while (isspace((unsigned char)*p)) p++;
        const char *ps = p;
        while (is_ident_char(*p)) p++;
        if (p > ps) { *out_parent = ps; *out_parent_len = (size_t)(p - ps); }
    }
    return 1;
}

static int match_type(const char *line, const char **out_name, size_t *out_name_len) {
    if (!starts_with(line, "type")) return 0;
    const char *p = line + 4;
    if (!isspace((unsigned char)*p)) return 0;
    while (isspace((unsigned char)*p)) p++;
    const char *ns = p;
    while (is_ident_char(*p)) p++;
    if (p == ns) return 0;
    *out_name = ns;
    *out_name_len = (size_t)(p - ns);
    return 1;
}

static int match_const(const char *line, const char **out_name, size_t *out_name_len,
                       const char **out_value, size_t *out_value_len) {
    if (!starts_with(line, "const")) return 0;
    const char *p = line + 5;
    if (!isspace((unsigned char)*p)) return 0;
    while (isspace((unsigned char)*p)) p++;
    const char *ns = p;
    while (is_ident_char(*p)) p++;
    if (p == ns) return 0;
    *out_name = ns;
    *out_name_len = (size_t)(p - ns);
    while (isspace((unsigned char)*p)) p++;
    if (*p != '=') return 0;
    p++;
    while (isspace((unsigned char)*p)) p++;
    *out_value = p;
    *out_value_len = strlen(p);

    while (*out_value_len > 0) {
        char c = (*out_value)[*out_value_len - 1];
        if (c == ' ' || c == '\t' || c == '\r') (*out_value_len)--;
        else break;
    }
    return 1;
}

typedef struct {
    const char *name;
    const char *kind;
    const char *deflt;
} ParamInfo;

typedef struct {
    const char *name;
    ParamInfo  *params;
    size_t      params_len;
    const char *doc;
    int         line;
} FnInfo;

typedef struct {
    const char *name;
    const char *deflt;
} ClassField;

typedef struct {
    const char *name;
    const char *parent;
    ClassField *fields;
    size_t      fields_len;
    FnInfo     *methods;
    size_t      methods_len;
    const char *doc;
    int         line;
} ClassInfo;

typedef struct {
    const char *name;
    const char **fields;
    size_t       fields_len;
    const char *doc;
    int         line;
} TypeInfo;

typedef struct {
    const char *name;
    const char *value;
} ConstInfo;

typedef struct {
    const char *module_doc;
    FnInfo     *functions;
    size_t      functions_len;
    ClassInfo  *classes;
    size_t      classes_len;
    TypeInfo   *types;
    size_t      types_len;
    ConstInfo  *constants;
    size_t      constants_len;
    const char **imports;
    size_t       imports_len;
} DocModel;

typedef struct { void *items; size_t len, cap, esize; } Vec;
static void vec_init(Vec *v, size_t es) { v->items=NULL; v->len=v->cap=0; v->esize=es; }
static void *vec_push(Vec *v) {
    if (v->len + 1 > v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 8;
        v->items = realloc(v->items, nc * v->esize);
        v->cap = nc;
    }
    void *out = (char *)v->items + v->len * v->esize;
    memset(out, 0, v->esize);
    v->len++;
    return out;
}
static void *vec_to_arena(TriadArena *a, Vec *v) {
    if (v->len == 0) { free(v->items); v->items = NULL; return NULL; }
    void *out = triad_arena_alloc(a, v->len * v->esize);
    memcpy(out, v->items, v->len * v->esize);
    free(v->items); v->items = NULL;
    return out;
}

static void parse_params(TriadArena *a, const char *params_raw,
                         size_t raw_len, ParamInfo **out, size_t *out_len) {

    while (raw_len && (*params_raw == ' ' || *params_raw == '\t')) {
        params_raw++; raw_len--;
    }
    while (raw_len && (params_raw[raw_len-1] == ' ' || params_raw[raw_len-1] == '\t'))
        raw_len--;
    if (raw_len == 0) { *out = NULL; *out_len = 0; return; }

    Vec v; vec_init(&v, sizeof(ParamInfo));
    const char *p = params_raw;
    const char *end = params_raw + raw_len;
    while (p < end) {
        const char *seg = p;
        while (p < end && *p != ',') p++;
        size_t L = (size_t)(p - seg);
        if (p < end && *p == ',') p++;

        while (L > 0 && (*seg == ' ' || *seg == '\t')) { seg++; L--; }
        while (L > 0 && (seg[L-1] == ' ' || seg[L-1] == '\t')) L--;
        if (L == 0) continue;

        ParamInfo *pi = (ParamInfo *)vec_push(&v);
        if (L >= 2 && seg[0] == '*' && seg[1] == '*') {
            char *nm = (char *)triad_arena_alloc(a, L - 1);
            memcpy(nm, seg + 2, L - 2); nm[L-2] = '\0';
            pi->name = nm; pi->kind = "kwargs"; pi->deflt = NULL;
        } else if (L >= 1 && seg[0] == '*') {
            char *nm = (char *)triad_arena_alloc(a, L);
            memcpy(nm, seg + 1, L - 1); nm[L-1] = '\0';
            pi->name = nm; pi->kind = "args"; pi->deflt = NULL;
        } else {

            const char *eq = NULL;
            for (size_t i = 0; i < L; ++i) {
                if (seg[i] == '=') { eq = seg + i; break; }
            }
            if (eq) {
                size_t nl = (size_t)(eq - seg);
                while (nl > 0 && (seg[nl-1] == ' ' || seg[nl-1] == '\t')) nl--;
                char *nm = (char *)triad_arena_alloc(a, nl + 1);
                memcpy(nm, seg, nl); nm[nl] = '\0';
                pi->name = nm;
                const char *dv = eq + 1;
                size_t dl = L - (size_t)(eq - seg) - 1;
                while (dl > 0 && (*dv == ' ' || *dv == '\t')) { dv++; dl--; }
                while (dl > 0 && (dv[dl-1] == ' ' || dv[dl-1] == '\t')) dl--;
                char *df = (char *)triad_arena_alloc(a, dl + 1);
                memcpy(df, dv, dl); df[dl] = '\0';
                pi->deflt = df;
                pi->kind = NULL;
            } else {
                char *nm = (char *)triad_arena_alloc(a, L + 1);
                memcpy(nm, seg, L); nm[L] = '\0';
                pi->name = nm; pi->kind = NULL; pi->deflt = NULL;
            }
        }
    }
    *out_len = v.len;
    *out = (ParamInfo *)vec_to_arena(a, &v);
}

static int parse_fn_at(TriadArena *a, const Lines *L, size_t start, FnInfo *out) {
    const char *line = lstrip(L->lines[start]);
    const char *name; size_t nlen;
    const char *params; size_t plen;
    if (!match_fn(line, &name, &nlen, &params, &plen)) return 0;
    char *nm = (char *)triad_arena_alloc(a, nlen + 1);
    memcpy(nm, name, nlen); nm[nlen] = '\0';
    out->name = nm;
    parse_params(a, params, plen, &out->params, &out->params_len);
    out->doc = "";
    if (start + 1 < L->n) {
        const char *next = lstrip(L->lines[start + 1]);
        if (starts_with(next, "//")) {
            out->doc = strip_dup(a, next + 2);
        }
    }
    out->line = (int)(start + 1);
    return 1;
}

static int parse_class_at(TriadArena *a, const Lines *L, size_t start, ClassInfo *out) {
    const char *line = lstrip(L->lines[start]);
    const char *name; size_t nlen;
    const char *parent; size_t plen;
    if (!match_class(line, &name, &nlen, &parent, &plen)) return 0;
    char *nm = (char *)triad_arena_alloc(a, nlen + 1);
    memcpy(nm, name, nlen); nm[nlen] = '\0';
    out->name = nm;
    if (parent) {
        char *pp = (char *)triad_arena_alloc(a, plen + 1);
        memcpy(pp, parent, plen); pp[plen] = '\0';
        out->parent = pp;
    } else {
        out->parent = "";
    }
    out->doc = "";
    out->line = (int)(start + 1);

    Vec fields; vec_init(&fields, sizeof(ClassField));
    Vec methods; vec_init(&methods, sizeof(FnInfo));

    size_t i = start + 1;
    while (i < L->n) {
        const char *cline = L->lines[i];
        int cind = leading_spaces(cline);
        const char *cstr = cline + cind;
        if (cind == 0 && *cstr) break;
        if (*cstr == '\0') { i++; continue; }
        if (starts_with(cstr, "//") && out->doc[0] == '\0') {
            out->doc = strip_dup(a, cstr + 2);
            i++; continue;
        }
        if (starts_with(cstr, "fn ") || starts_with(cstr, "fn\t")) {
            const char *fn_name; size_t fnl;
            const char *fn_params; size_t fpl;
            if (match_fn(cstr, &fn_name, &fnl, &fn_params, &fpl)) {
                FnInfo *m = (FnInfo *)vec_push(&methods);
                char *nm2 = (char *)triad_arena_alloc(a, fnl + 1);
                memcpy(nm2, fn_name, fnl); nm2[fnl] = '\0';
                m->name = nm2;
                parse_params(a, fn_params, fpl, &m->params, &m->params_len);
                m->doc = "";
                m->line = 0;
            }
            i++; continue;
        }

        const char *q = cstr;
        const char *id_start = q;
        while (is_ident_char(*q)) q++;
        if (q == id_start) { i++; continue; }
        size_t id_len = (size_t)(q - id_start);
        const char *r = q;
        while (*r == ' ' || *r == '\t') r++;
        if (*r == '\0' || *r == '=') {
            ClassField *fl = (ClassField *)vec_push(&fields);
            char *nm2 = (char *)triad_arena_alloc(a, id_len + 1);
            memcpy(nm2, id_start, id_len); nm2[id_len] = '\0';
            fl->name = nm2;
            fl->deflt = NULL;
            if (*r == '=') {
                r++;
                while (*r == ' ' || *r == '\t') r++;
                const char *de = r + strlen(r);
                while (de > r && (de[-1] == ' ' || de[-1] == '\t' || de[-1] == '\r')) de--;
                size_t dl = (size_t)(de - r);
                if (dl > 0) {
                    char *df = (char *)triad_arena_alloc(a, dl + 1);
                    memcpy(df, r, dl); df[dl] = '\0';
                    fl->deflt = df;
                }
            }
        }
        i++;
    }
    out->fields_len = fields.len;
    out->fields = (ClassField *)vec_to_arena(a, &fields);
    out->methods_len = methods.len;
    out->methods = (FnInfo *)vec_to_arena(a, &methods);
    return 1;
}

static int parse_type_at(TriadArena *a, const Lines *L, size_t start, TypeInfo *out) {
    const char *line = lstrip(L->lines[start]);
    const char *name; size_t nlen;
    if (!match_type(line, &name, &nlen)) return 0;
    char *nm = (char *)triad_arena_alloc(a, nlen + 1);
    memcpy(nm, name, nlen); nm[nlen] = '\0';
    out->name = nm;
    out->doc = "";
    out->line = (int)(start + 1);

    Vec fields; vec_init(&fields, sizeof(char *));

    size_t i = start + 1;
    while (i < L->n) {
        const char *cline = L->lines[i];
        int cind = leading_spaces(cline);
        const char *cstr = cline + cind;
        if (cind == 0 && *cstr) break;
        if (*cstr == '\0') { i++; continue; }
        if (starts_with(cstr, "//") && out->doc[0] == '\0') {
            out->doc = strip_dup(a, cstr + 2);
        } else {

            const char *p = cline;
            while (*p == ' ' || *p == '\t') p++;
            if (p > cline) {
                const char *ids = p;
                while (is_ident_char(*p)) p++;
                if (p > ids) {
                    size_t il = (size_t)(p - ids);
                    char *nm2 = (char *)triad_arena_alloc(a, il + 1);
                    memcpy(nm2, ids, il); nm2[il] = '\0';
                    const char **slot = (const char **)vec_push(&fields);
                    *slot = nm2;
                }
            }
        }
        i++;
    }
    out->fields_len = fields.len;
    out->fields = (const char **)vec_to_arena(a, &fields);
    return 1;
}

static DocModel parse_tri_docs(TriadArena *a, const char *source) {
    DocModel m = {0};
    m.module_doc = "";
    Lines L = split_lines(a, source);

    Vec functions; vec_init(&functions, sizeof(FnInfo));
    Vec classes;   vec_init(&classes, sizeof(ClassInfo));
    Vec types;     vec_init(&types, sizeof(TypeInfo));
    Vec consts;    vec_init(&consts, sizeof(ConstInfo));
    Vec imports;   vec_init(&imports, sizeof(char *));

    size_t i = 0;
    while (i < L.n) {
        const char *line = L.lines[i];
        int indent = leading_spaces(line);
        const char *str = line + indent;

        if (starts_with(str, "//")) {
            const char *cm = strip_dup(a, str + 2);
            if (m.module_doc[0] == '\0' && indent == 0) m.module_doc = cm;
            i++;
            continue;
        }

        if (indent == 0 && starts_with(str, "fn ")) {
            FnInfo fi = {0};
            if (parse_fn_at(a, &L, i, &fi)) {
                FnInfo *slot = (FnInfo *)vec_push(&functions);
                *slot = fi;
            }
        }

        if (indent == 0 && starts_with(str, "class ")) {
            ClassInfo ci = {0};
            if (parse_class_at(a, &L, i, &ci)) {
                ClassInfo *slot = (ClassInfo *)vec_push(&classes);
                *slot = ci;
            }
        }

        if (indent == 0 && starts_with(str, "type ")) {
            TypeInfo ti = {0};
            if (parse_type_at(a, &L, i, &ti)) {
                TypeInfo *slot = (TypeInfo *)vec_push(&types);
                *slot = ti;
            }
        }

        if (indent == 0 && starts_with(str, "const ")) {
            const char *cn; size_t cnl;
            const char *cv; size_t cvl;
            if (match_const(str, &cn, &cnl, &cv, &cvl)) {
                ConstInfo *slot = (ConstInfo *)vec_push(&consts);
                char *nm = (char *)triad_arena_alloc(a, cnl + 1);
                memcpy(nm, cn, cnl); nm[cnl] = '\0';
                slot->name = nm;
                char *vv = (char *)triad_arena_alloc(a, cvl + 1);
                memcpy(vv, cv, cvl); vv[cvl] = '\0';
                slot->value = vv;
            }
        }

        if (indent == 0 && (starts_with(str, "import ") || starts_with(str, "from "))) {
            const char **slot = (const char **)vec_push(&imports);
            *slot = strip_dup(a, str);
        }

        i++;
    }

    m.functions_len = functions.len; m.functions = (FnInfo *)vec_to_arena(a, &functions);
    m.classes_len = classes.len;     m.classes = (ClassInfo *)vec_to_arena(a, &classes);
    m.types_len = types.len;         m.types = (TypeInfo *)vec_to_arena(a, &types);
    m.constants_len = consts.len;    m.constants = (ConstInfo *)vec_to_arena(a, &consts);
    m.imports_len = imports.len;     m.imports = (const char **)vec_to_arena(a, &imports);
    return m;
}

static void emit_param_str(Buf *b, const ParamInfo *p) {
    if (p->kind && strcmp(p->kind, "kwargs") == 0) {
        buf_printf(b, "**%s", p->name);
    } else if (p->kind && strcmp(p->kind, "args") == 0) {
        buf_printf(b, "*%s", p->name);
    } else if (p->deflt) {
        buf_printf(b, "%s=%s", p->name, p->deflt);
    } else {
        buf_puts(b, p->name);
    }
}

static const char *render_markdown(TriadArena *a, const DocModel *d, const char *filename) {
    Buf parts; buf_init(&parts);
    Buf tmp;

    #define APPEND(s) do { buf_puts(&parts, (s)); buf_putc(&parts, '\n'); } while(0)
    #define APPEND_NL() buf_putc(&parts, '\n')

    const char *title = (filename && filename[0]) ? filename : "Module";
    buf_init(&tmp);
    buf_printf(&tmp, "# %s\n", title);
    APPEND(tmp.data); buf_free(&tmp);

    if (d->module_doc[0]) {
        buf_init(&tmp);
        buf_printf(&tmp, "%s\n", d->module_doc);
        APPEND(tmp.data); buf_free(&tmp);
    }

    if (d->imports_len) {
        APPEND("## Imports\n");
        for (size_t i = 0; i < d->imports_len; ++i) {
            buf_init(&tmp);
            buf_printf(&tmp, "- `%s`", d->imports[i]);
            APPEND(tmp.data); buf_free(&tmp);
        }
        APPEND("");
    }

    if (d->constants_len) {
        APPEND("## Constants\n");
        for (size_t i = 0; i < d->constants_len; ++i) {
            buf_init(&tmp);
            buf_printf(&tmp, "- **`%s`** = `%s`", d->constants[i].name, d->constants[i].value);
            APPEND(tmp.data); buf_free(&tmp);
        }
        APPEND("");
    }

    if (d->types_len) {
        APPEND("## Types\n");
        for (size_t i = 0; i < d->types_len; ++i) {
            const TypeInfo *t = &d->types[i];
            buf_init(&tmp); buf_printf(&tmp, "### `%s`\n", t->name); APPEND(tmp.data); buf_free(&tmp);
            if (t->doc[0]) {
                buf_init(&tmp); buf_printf(&tmp, "%s\n", t->doc); APPEND(tmp.data); buf_free(&tmp);
            }
            if (t->fields_len) {
                buf_init(&tmp); buf_puts(&tmp, "Fields: ");
                for (size_t j = 0; j < t->fields_len; ++j) {
                    if (j) buf_puts(&tmp, ", ");
                    buf_printf(&tmp, "`%s`", t->fields[j]);
                }
                APPEND(tmp.data); buf_free(&tmp);
                APPEND("");
            }
        }
    }

    if (d->functions_len) {
        APPEND("## Functions\n");
        for (size_t i = 0; i < d->functions_len; ++i) {
            const FnInfo *fn = &d->functions[i];
            buf_init(&tmp);
            buf_printf(&tmp, "### `fn %s(", fn->name);
            for (size_t j = 0; j < fn->params_len; ++j) {
                if (j) buf_puts(&tmp, ", ");
                emit_param_str(&tmp, &fn->params[j]);
            }
            buf_puts(&tmp, ")`\n");
            APPEND(tmp.data); buf_free(&tmp);
            if (fn->doc[0]) {
                buf_init(&tmp); buf_printf(&tmp, "%s\n", fn->doc); APPEND(tmp.data); buf_free(&tmp);
            }
            buf_init(&tmp); buf_printf(&tmp, "*Defined at line %d*\n", fn->line);
            APPEND(tmp.data); buf_free(&tmp);
        }
    }

    if (d->classes_len) {
        APPEND("## Classes\n");
        for (size_t i = 0; i < d->classes_len; ++i) {
            const ClassInfo *cl = &d->classes[i];
            buf_init(&tmp);
            if (cl->parent[0])
                buf_printf(&tmp, "### `class %s < %s`\n", cl->name, cl->parent);
            else
                buf_printf(&tmp, "### `class %s`\n", cl->name);
            APPEND(tmp.data); buf_free(&tmp);
            if (cl->doc[0]) {
                buf_init(&tmp); buf_printf(&tmp, "%s\n", cl->doc); APPEND(tmp.data); buf_free(&tmp);
            }
            if (cl->fields_len) {
                buf_init(&tmp); buf_puts(&tmp, "**Fields:** ");
                for (size_t j = 0; j < cl->fields_len; ++j) {
                    if (j) buf_puts(&tmp, ", ");
                    buf_printf(&tmp, "`%s`", cl->fields[j].name);
                    if (cl->fields[j].deflt)
                        buf_printf(&tmp, " = `%s`", cl->fields[j].deflt);
                }
                APPEND(tmp.data); buf_free(&tmp);
                APPEND("");
            }
            if (cl->methods_len) {
                APPEND("**Methods:**\n");
                for (size_t j = 0; j < cl->methods_len; ++j) {
                    const FnInfo *mt = &cl->methods[j];
                    buf_init(&tmp);
                    buf_printf(&tmp, "- `fn %s(", mt->name);
                    for (size_t k = 0; k < mt->params_len; ++k) {
                        if (k) buf_puts(&tmp, ", ");
                        emit_param_str(&tmp, &mt->params[k]);
                    }
                    buf_puts(&tmp, ")`");
                    APPEND(tmp.data); buf_free(&tmp);
                }
                APPEND("");
            }
        }
    }

    #undef APPEND
    #undef APPEND_NL

    if (parts.len > 0 && parts.data[parts.len - 1] == '\n') {
        parts.len--;
        parts.data[parts.len] = '\0';
    }
    return buf_to_arena(a, &parts);
}

static void html_escape_into(Buf *b, const char *s) {
    for (; *s; ++s) {
        switch (*s) {
        case '&': buf_puts(b, "&amp;"); break;
        case '<': buf_puts(b, "&lt;"); break;
        case '>': buf_puts(b, "&gt;"); break;
        default:  buf_putc(b, *s);
        }
    }
}

static void inline_format_into(Buf *b, const char *s) {

    Buf escaped; buf_init(&escaped);
    html_escape_into(&escaped, s);

    Buf step1; buf_init(&step1);
    const char *p = escaped.data;
    while (*p) {
        if (*p == '`') {
            const char *q = p + 1;
            while (*q && *q != '`') q++;
            if (*q == '`' && q > p + 1) {
                buf_puts(&step1, "<code>");
                buf_append(&step1, p + 1, (size_t)(q - p - 1));
                buf_puts(&step1, "</code>");
                p = q + 1;
                continue;
            }
        }
        buf_putc(&step1, *p++);
    }
    buf_free(&escaped);

    p = step1.data;
    while (*p) {
        if (*p == '*' && p[1] == '*') {
            const char *q = p + 2;
            while (*q && *q != '*') q++;
            if (*q == '*' && q[1] == '*' && q > p + 2) {
                buf_puts(b, "<strong>");
                buf_append(b, p + 2, (size_t)(q - p - 2));
                buf_puts(b, "</strong>");
                p = q + 2;
                continue;
            }
        }
        buf_putc(b, *p++);
    }
    buf_free(&step1);
}

static const char *strip_stars(TriadArena *a, const char *s) {
    while (*s == '*') s++;
    size_t L = strlen(s);
    while (L > 0 && s[L-1] == '*') L--;
    char *o = (char *)triad_arena_alloc(a, L + 1);
    memcpy(o, s, L); o[L] = '\0';
    return o;
}

static const char *render_html(TriadArena *a, const DocModel *d, const char *filename) {
    const char *md = render_markdown(a, d, filename);

    Buf out; buf_init(&out);
    buf_puts(&out, "<!DOCTYPE html>\n");
    buf_puts(&out, "<html><head><meta charset='utf-8'>\n");
    buf_printf(&out, "<title>%s</title>\n",
               (filename && filename[0]) ? filename : "TriadLang Docs");
    buf_puts(&out,
        "<style>body{font-family:system-ui;max-width:800px;margin:2em auto;padding:0 1em;}"
        "code{background:#f4f4f4;padding:2px 6px;border-radius:3px;font-size:0.9em;}"
        "pre{background:#f4f4f4;padding:1em;border-radius:6px;overflow-x:auto;}"
        "h1{border-bottom:2px solid #333;padding-bottom:0.3em;}"
        "h2{border-bottom:1px solid #999;padding-bottom:0.2em;margin-top:2em;}"
        "h3{color:#555;}</style></head><body>\n");

    int in_code = 0;

    Lines L = split_lines(a, md);
    for (size_t i = 0; i < L.n; ++i) {
        const char *line = L.lines[i];
        if (starts_with(line, "```")) {
            if (in_code) buf_puts(&out, "</pre>\n");
            else         buf_puts(&out, "<pre><code>\n");
            in_code = !in_code;
            continue;
        }
        if (in_code) {
            html_escape_into(&out, line);
            buf_putc(&out, '\n');
            continue;
        }
        if (starts_with(line, "# ")) {
            buf_puts(&out, "<h1>");
            html_escape_into(&out, line + 2);
            buf_puts(&out, "</h1>\n");
        } else if (starts_with(line, "## ")) {
            buf_puts(&out, "<h2>");
            html_escape_into(&out, line + 3);
            buf_puts(&out, "</h2>\n");
        } else if (starts_with(line, "### ")) {
            buf_puts(&out, "<h3>");
            html_escape_into(&out, line + 4);
            buf_puts(&out, "</h3>\n");
        } else if (starts_with(line, "- ")) {
            buf_puts(&out, "<li>");
            inline_format_into(&out, line + 2);
            buf_puts(&out, "</li>\n");
        } else {

            size_t L_ = strlen(line);
            if (L_ >= 2 && line[0] == '*' && line[L_-1] == '*') {
                const char *stripped = strip_stars(a, line);
                buf_puts(&out, "<p><em>");
                html_escape_into(&out, stripped);
                buf_puts(&out, "</em></p>\n");
            } else {

                int has = 0;
                for (const char *p = line; *p; ++p) {
                    if (*p != ' ' && *p != '\t') { has = 1; break; }
                }
                if (has) {
                    buf_puts(&out, "<p>");
                    inline_format_into(&out, line);
                    buf_puts(&out, "</p>\n");
                } else {
                    buf_putc(&out, '\n');
                }
            }
        }
    }
    buf_puts(&out, "</body></html>");

    return buf_to_arena(a, &out);
}

const char *triad_docgen_markdown(TriadArena *arena,
                                  const char *source,
                                  const char *filename) {
    if (!arena || !source) return NULL;
    DocModel d = parse_tri_docs(arena, source);
    return render_markdown(arena, &d, filename ? filename : "");
}

const char *triad_docgen_html(TriadArena *arena,
                              const char *source,
                              const char *filename) {
    if (!arena || !source) return NULL;
    DocModel d = parse_tri_docs(arena, source);
    return render_html(arena, &d, filename ? filename : "");
}
