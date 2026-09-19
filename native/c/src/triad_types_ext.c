#include "triad_rt.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

TriadTuple *triad_tuple_new(int32_t len) {
    if (len < 0) return NULL;
    TriadTuple *t = calloc(1, sizeof(TriadTuple));
    if (!t) return NULL;
    t->refcount = 1;
    t->len = len;
    t->items = len > 0 ? calloc((size_t)len, sizeof(TriadValue)) : NULL;
    if (len > 0 && !t->items) { free(t); return NULL; }
    if (t->items) {
        for (int32_t i = 0; i < len; i++) t->items[i] = TRIAD_NONE_VAL;
    }
    return t;
}

void triad_tuple_free(TriadTuple *t) {
    if (!t) return;
    if (t->items) free(t->items);
    free(t);
}

TriadValue triad_tuple_get(TriadTuple *t, int32_t idx) {
    if (!t || idx < 0 || idx >= t->len) return TRIAD_NONE_VAL;
    return t->items[idx];
}

void triad_tuple_set(TriadTuple *t, int32_t idx, TriadValue v) {
    if (!t || idx < 0 || idx >= t->len) return;
    t->items[idx] = v;
}

int32_t triad_tuple_len(TriadTuple *t) {
    return t ? t->len : 0;
}

bool triad_tuple_eq(TriadTuple *a, TriadTuple *b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    if (a->len != b->len) return false;
    for (int32_t i = 0; i < a->len; i++) {
        if (!triad_value_eq(a->items[i], b->items[i])) return false;
    }
    return true;
}

int32_t triad_tuple_hash(TriadTuple *t) {
    if (!t) return 0;
    int32_t h = 0x9e3779b9;
    for (int32_t i = 0; i < t->len; i++) {
        TriadValue v = t->items[i];
        h ^= (v.tag << (i & 7));
        if (v.tag == TRIAD_INT) h ^= (int32_t)(v.as.ival * 2654435761u);
        else if (v.tag == TRIAD_FLOAT) h ^= (int32_t)(v.as.fval * 2654435761u);
        else if (v.tag == TRIAD_STRING && v.as.sval) h ^= triad_str_hash(v.as.sval);
    }
    return h;
}

TriadBytes *triad_bytes_new(const uint8_t *data, int32_t len) {
    if (len < 0) return NULL;
    TriadBytes *b = calloc(1, sizeof(TriadBytes));
    if (!b) return NULL;
    b->refcount = 1;
    b->len = len;
    b->data = len > 0 ? malloc((size_t)len) : NULL;
    if (len > 0 && !b->data) { free(b); return NULL; }
    if (b->data && data) memcpy(b->data, data, (size_t)len);
    return b;
}

void triad_bytes_free(TriadBytes *b) {
    if (!b) return;
    if (b->data) free(b->data);
    free(b);
}

int32_t triad_bytes_len(TriadBytes *b) {
    return b ? b->len : 0;
}

bool triad_bytes_eq(TriadBytes *a, TriadBytes *b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    if (a->len != b->len) return false;
    return memcmp(a->data, b->data, a->len) == 0;
}

int32_t triad_bytes_hash(TriadBytes *b) {
    if (!b) return 0;
    int32_t h = 0x9e3779b9;
    for (int32_t i = 0; i < b->len; i++) h = h * 31 + b->data[i];
    return h;
}

TriadString *triad_bytes_repr(TriadBytes *b) {
    if (!b) return triad_str_new("b''");
    int32_t cap = b->len * 4 + 4;
    char *out = malloc(cap);
    int32_t pos = 0;
    out[pos++] = 'b';
    out[pos++] = '\'';
    for (int32_t i = 0; i < b->len && pos < cap - 4; i++) {
        uint8_t c = b->data[i];
        if (c >= 32 && c < 127 && c != '\\' && c != '\'') {
            out[pos++] = c;
        } else {
            pos += snprintf(out + pos, cap - pos, "\\x%02x", c);
        }
    }
    out[pos++] = '\'';
    out[pos] = '\0';
    TriadString *s = triad_str_new(out);
    free(out);
    return s;
}

TriadComplex *triad_complex_new(double re, double im) {
    TriadComplex *c = calloc(1, sizeof(TriadComplex));
    if (!c) return NULL;
    c->refcount = 1;
    c->re = re;
    c->im = im;
    return c;
}

void triad_complex_free(TriadComplex *c) {
    if (c) free(c);
}

double triad_complex_abs(TriadComplex *c) {
    if (!c) return 0.0;
    return sqrt(c->re * c->re + c->im * c->im);
}

TriadComplex *triad_complex_add(TriadComplex *a, TriadComplex *b) {
    if (!a || !b) return NULL;
    return triad_complex_new(a->re + b->re, a->im + b->im);
}

TriadComplex *triad_complex_sub(TriadComplex *a, TriadComplex *b) {
    if (!a || !b) return NULL;
    return triad_complex_new(a->re - b->re, a->im - b->im);
}

TriadComplex *triad_complex_mul(TriadComplex *a, TriadComplex *b) {
    if (!a || !b) return NULL;
    return triad_complex_new(a->re * b->re - a->im * b->im,
                             a->re * b->im + a->im * b->re);
}

TriadComplex *triad_complex_div(TriadComplex *a, TriadComplex *b) {
    if (!a || !b) return NULL;
    double denom = b->re * b->re + b->im * b->im;
    if (denom == 0.0) return NULL;
    return triad_complex_new((a->re * b->re + a->im * b->im) / denom,
                             (a->im * b->re - a->re * b->im) / denom);
}

bool triad_complex_eq(TriadComplex *a, TriadComplex *b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return a->re == b->re && a->im == b->im;
}

TriadString *triad_complex_repr(TriadComplex *c) {
    char buf[128];
    if (!c) return triad_str_new("0j");
    if (c->im == 0.0) {
        snprintf(buf, sizeof(buf), "(%g+0j)", c->re);
    } else if (c->re == 0.0) {
        snprintf(buf, sizeof(buf), "%gj", c->im);
    } else {
        snprintf(buf, sizeof(buf), "(%g%+gj)", c->re, c->im);
    }
    return triad_str_new(buf);
}

#define TRIAD_SET_INIT_CAP 8

static int32_t _set_hash_value(TriadValue v) {
    switch (v.tag) {
    case TRIAD_INT:     return (int32_t)(v.as.ival * 2654435761u);
    case TRIAD_FLOAT:   return (int32_t)(v.as.fval * 2654435761u);
    case TRIAD_BOOL:    return v.as.bval ? 1 : 0;
    case TRIAD_STRING:  return v.as.sval ? triad_str_hash(v.as.sval) : 0;
    case TRIAD_TUPLE:   return triad_tuple_hash(v.as.tval);
    case TRIAD_BYTES:   return triad_bytes_hash(v.as.bvalp);
    default:            return (int32_t)(v.as.ival * 31);
    }
}

TriadSet *triad_set_new(void) {
    TriadSet *s = calloc(1, sizeof(TriadSet));
    if (!s) return NULL;
    s->refcount = 1;
    s->len = 0;
    s->cap = TRIAD_SET_INIT_CAP;
    s->entries = calloc((size_t)s->cap, sizeof(TriadSetEntry));
    if (!s->entries) { free(s); return NULL; }
    return s;
}

void triad_set_free(TriadSet *s) {
    if (!s) return;
    if (s->entries) {
        for (int32_t i = 0; i < s->cap; i++) {
            if (s->entries[i].used) triad_release(&s->entries[i].key);
        }
        free(s->entries);
    }
    free(s);
}

static void _set_grow(TriadSet *s) {
    if (!s || !s->entries || s->cap < 1) return;
    int32_t old_cap = s->cap;
    TriadSetEntry *old = s->entries;
    TriadSetEntry *fresh = calloc((size_t)old_cap * 2, sizeof(TriadSetEntry));
    if (!fresh) return;
    s->cap = old_cap * 2;
    s->entries = fresh;
    s->len = 0;
    for (int32_t i = 0; i < old_cap; i++) {
        if (old[i].used) {
            triad_set_add(s, old[i].key);
        }
    }
    free(old);
}

void triad_set_add(TriadSet *s, TriadValue v) {
    if (!s || !s->entries || s->cap < 1) return;
    if (s->len * 2 >= s->cap) _set_grow(s);
    if (!s->entries || s->cap < 1) return;
    int32_t h = _set_hash_value(v);
    int32_t idx = h & (s->cap - 1);
    for (int32_t i = 0; i < s->cap; i++) {
        int32_t slot = (idx + i) & (s->cap - 1);
        if (!s->entries[slot].used) {
            s->entries[slot].key = v;
            s->entries[slot].hash = h;
            s->entries[slot].used = true;
            s->len++;
            return;
        }
        if (s->entries[slot].hash == h && triad_value_eq(s->entries[slot].key, v)) {

            return;
        }
    }
}

bool triad_set_has(TriadSet *s, TriadValue v) {
    if (!s || !s->entries || s->cap < 1) return false;
    int32_t h = _set_hash_value(v);
    int32_t idx = h & (s->cap - 1);
    for (int32_t i = 0; i < s->cap; i++) {
        int32_t slot = (idx + i) & (s->cap - 1);
        if (!s->entries[slot].used) return false;
        if (s->entries[slot].hash == h && triad_value_eq(s->entries[slot].key, v))
            return true;
    }
    return false;
}

int32_t triad_set_len(TriadSet *s) {
    return s ? s->len : 0;
}

TriadList *triad_set_to_list(TriadSet *s) {
    TriadList *l = triad_list_new_cap(s ? s->len : 0);
    if (s) {
        for (int32_t i = 0; i < s->cap; i++) {
            if (s->entries[i].used) triad_list_push(l, s->entries[i].key);
        }
    }
    return l;
}

bool triad_set_eq(TriadSet *a, TriadSet *b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    if (a->len != b->len) return false;
    for (int32_t i = 0; i < a->cap; i++) {
        if (a->entries[i].used && !triad_set_has(b, a->entries[i].key))
            return false;
    }
    return true;
}

TriadGenerator *triad_generator_new(void) {
    TriadGenerator *g = calloc(1, sizeof(TriadGenerator));
    if (!g) return NULL;
    g->refcount = 1;
    g->state = 0;
    g->current = TRIAD_NONE_VAL;
    g->next_fn = NULL;
    g->user_data = NULL;
    return g;
}

void triad_generator_free(TriadGenerator *g) {
    if (!g) return;
    triad_release(&g->current);
    free(g);
}

TriadValue triad_generator_next(TriadGenerator *g) {
    if (!g || !g->next_fn || g->state < 0) {
        return TRIAD_NONE_VAL;
    }
    TriadValue v = g->next_fn(g);
    triad_release(&g->current);
    g->current = v;
    triad_retain(&g->current);
    return v;
}

bool triad_generator_done(TriadGenerator *g) {
    return !g || g->state < 0 || !g->next_fn;
}

TriadValue _triad_yield_value(TriadValue v) {

    return v;
}

TriadValue _triad_await_value(TriadValue v) {

    return v;
}

#include <string.h>

typedef struct _ClassMetaEntry {
    TriadClassMeta *meta;
    struct _ClassMetaEntry *next;
} _ClassMetaEntry;

static _ClassMetaEntry *_class_meta_registry = NULL;

TriadClassMeta *triad_class_meta_new(const char *name, TriadClassMeta *parent,
                                      const char **fields, int32_t fields_len) {
    TriadClassMeta *cm = (TriadClassMeta *)calloc(1, sizeof(TriadClassMeta));
    if (!cm) return NULL;
    if (!name) { free(cm); return NULL; }
    if (fields_len < 0 || (fields_len > 0 && !fields)) { free(cm); return NULL; }
    cm->name = name;
    cm->parent = parent;
    cm->fields = fields;
    cm->fields_len = fields_len;
    cm->methods = NULL;
    cm->methods_len = 0;
    cm->refcount = 1;
    return cm;
}

void triad_class_meta_free(TriadClassMeta *cm) {
    if (!cm) return;
    cm->refcount--;
    if (cm->refcount <= 0) {
        if (cm->methods) free(cm->methods);
        free(cm);
    }
}

void triad_class_meta_register(TriadClassMeta *cm) {
    if (!cm) return;
    _ClassMetaEntry *entry = (_ClassMetaEntry *)calloc(1, sizeof(_ClassMetaEntry));
    if (!entry) return;
    entry->meta = cm;
    entry->next = _class_meta_registry;
    _class_meta_registry = entry;
}

TriadClassMeta *triad_class_meta_lookup(const char *name) {
    _ClassMetaEntry *cur = _class_meta_registry;
    if (!name) return NULL;
    while (cur) {
        if (cur->meta && cur->meta->name && strcmp(cur->meta->name, name) == 0)
            return cur->meta;
        cur = cur->next;
    }
    return NULL;
}

TriadNativeFn triad_class_meta_resolve_method(TriadClassMeta *cm, const char *method) {
    if (!cm || !method) return NULL;

    TriadClassMeta *cur = cm;
    while (cur) {
        for (int32_t i = 0; i < cur->methods_len; i++) {
            if (cur->methods[i].name && strcmp(cur->methods[i].name, method) == 0)
                return cur->methods[i].fn;
        }
        cur = cur->parent;
    }
    return NULL;
}

TriadClassMeta *triad_class_meta_get_parent(TriadClassMeta *cm) {
    return cm ? cm->parent : NULL;
}

TriadValue triad_object_new_typed(const char *type_name, TriadClassMeta *cm) {
    TriadObject *o = triad_object_new(type_name);
    if (!o) return TRIAD_NONE_VAL;
    o->class_meta = cm;
    return (TriadValue){.tag = TRIAD_OBJECT, .as = {.oval = o}};
}

TriadValue _triad_super(int32_t nargs, TriadValue *args) {

    if (nargs >= 2 && args && args[0].tag == TRIAD_OBJECT && args[1].tag == TRIAD_STRING) {
        TriadObject *self_obj = args[0].as.oval;
        if (!self_obj || !args[1].as.sval || !args[1].as.sval->data) return TRIAD_NONE_VAL;
        const char *cls_name = args[1].as.sval->data;
        TriadClassMeta *cm = triad_class_meta_lookup(cls_name);
        if (cm && cm->parent) {


            return (TriadValue){.tag = TRIAD_OBJECT, .as = {.oval = self_obj}};
        }
    }
    return TRIAD_NONE_VAL;
}
