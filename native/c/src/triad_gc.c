#include "triad_rt.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

jmp_buf   triad_jmp_bufs[TRIAD_MAX_JMP];
int32_t   triad_jmp_depth = 0;
TriadValue triad_exception = {.tag = TRIAD_NONE, .as = {.ival = 0}};

void triad_runtime_init(void) {
#if TRIAD_GC_BOEHM
    GC_INIT();
#endif
    triad_jmp_depth = 0;
    triad_exception.tag = TRIAD_NONE;
    triad_exception.as.ival = 0;
}

void triad_retain(TriadValue *v) {
    if (!v) return;
    switch (v->tag) {
    case TRIAD_STRING:  if (v->as.sval) v->as.sval->refcount++; break;
    case TRIAD_LIST:    if (v->as.lval) v->as.lval->refcount++; break;
    case TRIAD_DICT:    if (v->as.dval) v->as.dval->refcount++; break;
    case TRIAD_NDARRAY: if (v->as.aval) v->as.aval->refcount++; break;
    case TRIAD_CLOSURE: if (v->as.cval) v->as.cval->refcount++; break;
    case TRIAD_ITER:    if (v->as.itval) v->as.itval->refcount++; break;
    case TRIAD_OBJECT:  if (v->as.oval) v->as.oval->refcount++; break;
    case TRIAD_TUPLE:   if (v->as.tval) v->as.tval->refcount++; break;
    case TRIAD_BYTES:   if (v->as.bvalp) v->as.bvalp->refcount++; break;
    case TRIAD_COMPLEX: if (v->as.cvalp) v->as.cvalp->refcount++; break;
    case TRIAD_SET:     if (v->as.sset) v->as.sset->refcount++; break;
    case TRIAD_GENERATOR: if (v->as.gval) v->as.gval->refcount++; break;
    default: break;
    }
}

static void release_str(TriadString *s) {
    if (!s) return;
    s->refcount--;
    if (s->refcount <= 0) triad_str_free(s);
}

static void release_list(TriadList *l) {
    if (!l) return;
    l->refcount--;
    if (l->refcount <= 0) {
        for (int32_t i = 0; i < l->len; i++)
            triad_release(&l->items[i]);
        triad_list_free(l);
    }
}

static void release_dict(TriadDict *d) {
    if (!d) return;
    d->refcount--;
    if (d->refcount <= 0) {
        for (int32_t i = 0; i < d->cap; i++) {
            if (d->entries[i].key) {
                release_str(d->entries[i].key);
                triad_release(&d->entries[i].value);
            }
        }
        triad_dict_free(d);
    }
}

static void release_ndarray(TriadNDArray *a) {
    if (!a) return;
    a->refcount--;
    if (a->refcount <= 0) triad_ndarray_free(a);
}

static void release_closure(TriadClosure *c) {
    if (!c) return;
    c->refcount--;
    if (c->refcount <= 0) {
        for (int32_t i = 0; i < c->ncaptured; i++)
            triad_release(&c->captured[i]);
        triad_closure_free(c);
    }
}

static void release_iter(TriadIter *it) {
    if (!it) return;
    it->refcount--;
    if (it->refcount <= 0) {
        triad_release(&it->current);
        free(it);
    }
}

static void release_object(TriadObject *o) {
    if (!o) return;
    o->refcount--;
    if (o->refcount <= 0) {
        if (o->type_name) release_str(o->type_name);
        if (o->fields) release_dict(o->fields);
        free(o);
    }
}

static void release_tuple(TriadTuple *t) {
    if (!t) return;
    t->refcount--;
    if (t->refcount <= 0) {
        for (int32_t i = 0; i < t->len; i++)
            triad_release(&t->items[i]);
        free(t->items);
        free(t);
    }
}

static void release_bytes(TriadBytes *b) {
    if (!b) return;
    b->refcount--;
    if (b->refcount <= 0) {
        free(b->data);
        free(b);
    }
}

static void release_complex(TriadComplex *c) {
    if (!c) return;
    c->refcount--;
    if (c->refcount <= 0) free(c);
}

static void release_set(TriadSet *s) {
    if (!s) return;
    s->refcount--;
    if (s->refcount <= 0) {
        for (int32_t i = 0; i < s->cap; i++) {
            if (s->entries[i].used) triad_release(&s->entries[i].key);
        }
        free(s->entries);
        free(s);
    }
}

static void release_generator(TriadGenerator *g) {
    if (!g) return;
    g->refcount--;
    if (g->refcount <= 0) {
        triad_release(&g->current);
        free(g);
    }
}

void triad_release(TriadValue *v) {
    if (!v) return;
    switch (v->tag) {
    case TRIAD_STRING:  release_str(v->as.sval); break;
    case TRIAD_LIST:    release_list(v->as.lval); break;
    case TRIAD_DICT:    release_dict(v->as.dval); break;
    case TRIAD_NDARRAY: release_ndarray(v->as.aval); break;
    case TRIAD_CLOSURE: release_closure(v->as.cval); break;
    case TRIAD_ITER:    release_iter(v->as.itval); break;
    case TRIAD_OBJECT:  release_object(v->as.oval); break;
    case TRIAD_TUPLE:   release_tuple(v->as.tval); break;
    case TRIAD_BYTES:   release_bytes(v->as.bvalp); break;
    case TRIAD_COMPLEX: release_complex(v->as.cvalp); break;
    case TRIAD_SET:     release_set(v->as.sset); break;
    case TRIAD_GENERATOR: release_generator(v->as.gval); break;
    default: break;
    }
    v->tag = TRIAD_NONE;
    v->as.ival = 0;
}

void triad_throw(TriadValue exc) {
    triad_exception = exc;
    if (triad_jmp_depth > 0) {
        longjmp(triad_jmp_bufs[triad_jmp_depth - 1], 1);
    }
    TriadString *msg = triad_value_to_string(exc);
    fprintf(stderr, "Unhandled exception: %s\n", msg ? msg->data : "(unknown)");
    exit(1);
}

bool (*triad_pyobj_y_hook)(void *pyobj) = NULL;

bool triad_is_y(TriadValue v) {
    switch (v.tag) {
    case TRIAD_PYOBJ:
        if (triad_pyobj_y_hook != NULL)
            return triad_pyobj_y_hook(v.as.ptr);
        return true;
    case TRIAD_NONE:    return false;
    case TRIAD_BOOL:    return v.as.bval;
    case TRIAD_INT:     return v.as.ival != 0;
    case TRIAD_FLOAT:   return v.as.fval != 0.0;
    case TRIAD_STRING:  return v.as.sval && v.as.sval->len > 0;
    case TRIAD_LIST:    return v.as.lval && v.as.lval->len > 0;
    case TRIAD_DICT:    return v.as.dval && v.as.dval->len > 0;
    case TRIAD_NDARRAY: return v.as.aval && v.as.aval->size > 0;
    case TRIAD_TUPLE:   return v.as.tval && v.as.tval->len > 0;
    case TRIAD_BYTES:   return v.as.bvalp && v.as.bvalp->len > 0;
    case TRIAD_COMPLEX: return v.as.cvalp && (v.as.cvalp->re != 0.0 || v.as.cvalp->im != 0.0);
    case TRIAD_SET:     return v.as.sset && v.as.sset->len > 0;
    case TRIAD_GENERATOR: return true;
    default:            return true;
    }
}

bool triad_value_eq(TriadValue a, TriadValue b) {
    if (a.tag != b.tag) {

        if ((a.tag == TRIAD_INT || a.tag == TRIAD_FLOAT) &&
            (b.tag == TRIAD_INT || b.tag == TRIAD_FLOAT))
            return (a.tag == TRIAD_INT ? (double)a.as.ival : a.as.fval) ==
                   (b.tag == TRIAD_INT ? (double)b.as.ival : b.as.fval);
        return false;
    }
    switch (a.tag) {
    case TRIAD_NONE:    return true;
    case TRIAD_BOOL:    return a.as.bval == b.as.bval;
    case TRIAD_INT:     return a.as.ival == b.as.ival;
    case TRIAD_FLOAT:   return a.as.fval == b.as.fval;
    case TRIAD_STRING:  return triad_str_eq(a.as.sval, b.as.sval);
    case TRIAD_TUPLE:   return triad_tuple_eq(a.as.tval, b.as.tval);
    case TRIAD_BYTES:   return triad_bytes_eq(a.as.bvalp, b.as.bvalp);
    case TRIAD_COMPLEX: return triad_complex_eq(a.as.cvalp, b.as.cvalp);
    case TRIAD_SET:     return triad_set_eq(a.as.sset, b.as.sset);
    default:            return a.as.ival == b.as.ival;
    }
}

static const char *tag_name(TriadTag t) {
    switch (t) {
    case TRIAD_NONE:    return "none";
    case TRIAD_BOOL:    return "bool";
    case TRIAD_INT:     return "int";
    case TRIAD_FLOAT:   return "float";
    case TRIAD_STRING:  return "str";
    case TRIAD_LIST:    return "list";
    case TRIAD_DICT:    return "dict";
    case TRIAD_NDARRAY: return "ndarray";
    case TRIAD_CLOSURE: return "fn";
    case TRIAD_OBJECT:  return "object";
    case TRIAD_TUPLE:   return "tuple";
    case TRIAD_BYTES:   return "bytes";
    case TRIAD_COMPLEX: return "complex";
    case TRIAD_SET:     return "set";
    case TRIAD_GENERATOR: return "generator";
    default:            return "unknown";
    }
}

TriadValue triad_type_name(TriadValue v) {
    TriadString *s = triad_str_new(tag_name(v.tag));
    return (TriadValue){.tag = TRIAD_STRING, .as = {.sval = s}};
}

TriadString *(*triad_pyobj_str_hook)(void *pyobj) = NULL;

TriadString *triad_value_to_string(TriadValue v) {
    char buf[256];
    switch (v.tag) {
    case TRIAD_PYOBJ:
        if (triad_pyobj_str_hook != NULL)
            return triad_pyobj_str_hook(v.as.ptr);
        return triad_str_new("<pyobj>");
    case TRIAD_NONE:
        return triad_str_new("none");
    case TRIAD_BOOL:
        return triad_str_new(v.as.bval ? "true" : "false");
    case TRIAD_INT:
        snprintf(buf, sizeof(buf), "%lld", (long long)v.as.ival);
        return triad_str_new(buf);
    case TRIAD_FLOAT: {

        for (int prec = 1; prec <= 17; prec++) {
            snprintf(buf, sizeof(buf), "%.*g", prec, v.as.fval);
            if (strtod(buf, NULL) == v.as.fval) break;
        }

        if (strpbrk(buf, ".eEni") == NULL) {
            size_t l = strlen(buf);
            if (l + 2 < sizeof(buf)) { buf[l] = '.'; buf[l+1] = '0'; buf[l+2] = '\0'; }
        }
        return triad_str_new(buf);
    }
    case TRIAD_STRING:
        return triad_str_copy(v.as.sval);
    case TRIAD_LIST: {
        int32_t cap = 256;
        char *out = malloc(cap);
        int32_t pos = 0;
        out[pos++] = '[';
        for (int32_t i = 0; i < v.as.lval->len; i++) {
            if (i > 0) { out[pos++] = ','; out[pos++] = ' '; }
            TriadValue item = v.as.lval->items[i];

            TriadString *elem = triad_value_to_string(item);
            int quote = (item.tag == TRIAD_STRING);
            int32_t needed = pos + elem->len + 8;
            if (needed > cap) { cap = needed * 2; out = realloc(out, cap); }
            if (quote) out[pos++] = '"';
            memcpy(out + pos, elem->data, elem->len);
            pos += elem->len;
            if (quote) out[pos++] = '"';
            release_str(elem);
        }
        out[pos++] = ']';
        out[pos] = '\0';
        TriadString *s = triad_str_new(out);
        free(out);
        return s;
    }
    case TRIAD_TUPLE: {
        int32_t cap = 256;
        char *out = malloc(cap);
        int32_t pos = 0;
        out[pos++] = '(';
        for (int32_t i = 0; i < v.as.tval->len; i++) {
            if (i > 0) { out[pos++] = ','; out[pos++] = ' '; }
            TriadString *elem = triad_value_to_string(v.as.tval->items[i]);
            int32_t needed = pos + elem->len + 4;
            if (needed > cap) { cap = needed * 2; out = realloc(out, cap); }
            memcpy(out + pos, elem->data, elem->len);
            pos += elem->len;
            release_str(elem);
        }
        if (v.as.tval->len == 1) { out[pos++] = ','; }
        out[pos++] = ')';
        out[pos] = '\0';
        TriadString *s = triad_str_new(out);
        free(out);
        return s;
    }
    case TRIAD_BYTES: {
        TriadString *r = triad_bytes_repr(v.as.bvalp);
        return r;
    }
    case TRIAD_COMPLEX: {
        TriadString *r = triad_complex_repr(v.as.cvalp);
        return r;
    }
    case TRIAD_SET: {
        int32_t cap = 256;
        char *out = malloc(cap);
        int32_t pos = 0;
        out[pos++] = '{';
        TriadList *sl = triad_set_to_list(v.as.sset);
        for (int32_t i = 0; i < sl->len; i++) {
            if (i > 0) { out[pos++] = ','; out[pos++] = ' '; }
            TriadString *elem = triad_value_to_string(sl->items[i]);
            int32_t needed = pos + elem->len + 4;
            if (needed > cap) { cap = needed * 2; out = realloc(out, cap); }
            memcpy(out + pos, elem->data, elem->len);
            pos += elem->len;
            release_str(elem);
        }
        out[pos++] = '}';
        out[pos] = '\0';
        TriadString *s = triad_str_new(out);
        free(out);
        return s;
    }
    case TRIAD_GENERATOR: {
        return triad_str_new("<generator>");
    }
    default: {
        snprintf(buf, sizeof(buf), "<%s>", tag_name(v.tag));
        return triad_str_new(buf);
    }
    }
}
