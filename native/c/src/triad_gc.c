/*
 * TriadLang Native Runtime — GC (reference counting + optional Boehm)
 *
 * When compiled with -DTRIAD_USE_BOEHM:
 *   - Allocation goes through GC_MALLOC / GC_MALLOC_ATOMIC
 *   - triad_free is a no-op (Boehm collects automatically)
 *   - Reference counting still runs but triad_free becomes a no-op,
 *     so the decrement path is harmless (refcounts are advisory).
 *
 * When compiled without the flag (default):
 *   - Standard malloc/free with manual reference counting.
 */
#include "triad_rt.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

jmp_buf   triad_jmp_bufs[TRIAD_MAX_JMP];
int32_t   triad_jmp_depth = 0;
TriadValue triad_exception = {.tag = TRIAD_NONE, .as = {.ival = 0}};

/* ── Runtime initialization (called by TriadMain or directly) ── */

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

bool triad_is_truthy(TriadValue v) {
    switch (v.tag) {
    case TRIAD_NONE:    return false;
    case TRIAD_BOOL:    return v.as.bval;
    case TRIAD_INT:     return v.as.ival != 0;
    case TRIAD_FLOAT:   return v.as.fval != 0.0;
    case TRIAD_STRING:  return v.as.sval && v.as.sval->len > 0;
    case TRIAD_LIST:    return v.as.lval && v.as.lval->len > 0;
    case TRIAD_DICT:    return v.as.dval && v.as.dval->len > 0;
    case TRIAD_NDARRAY: return v.as.aval && v.as.aval->size > 0;
    default:            return true;
    }
}

bool triad_value_eq(TriadValue a, TriadValue b) {
    if (a.tag != b.tag) return false;
    switch (a.tag) {
    case TRIAD_NONE:   return true;
    case TRIAD_BOOL:   return a.as.bval == b.as.bval;
    case TRIAD_INT:    return a.as.ival == b.as.ival;
    case TRIAD_FLOAT:  return a.as.fval == b.as.fval;
    case TRIAD_STRING: return triad_str_eq(a.as.sval, b.as.sval);
    default:           return false;
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
    default:            return "unknown";
    }
}

TriadValue triad_type_name(TriadValue v) {
    TriadString *s = triad_str_new(tag_name(v.tag));
    return (TriadValue){.tag = TRIAD_STRING, .as = {.sval = s}};
}

TriadString *triad_value_to_string(TriadValue v) {
    char buf[256];
    switch (v.tag) {
    case TRIAD_NONE:
        return triad_str_new("none");
    case TRIAD_BOOL:
        return triad_str_new(v.as.bval ? "true" : "false");
    case TRIAD_INT:
        snprintf(buf, sizeof(buf), "%lld", (long long)v.as.ival);
        return triad_str_new(buf);
    case TRIAD_FLOAT:
        snprintf(buf, sizeof(buf), "%g", v.as.fval);
        return triad_str_new(buf);
    case TRIAD_STRING:
        return triad_str_copy(v.as.sval);
    case TRIAD_LIST: {
        int32_t cap = 256;
        char *out = malloc(cap);
        int32_t pos = 0;
        out[pos++] = '[';
        for (int32_t i = 0; i < v.as.lval->len; i++) {
            if (i > 0) { out[pos++] = ','; out[pos++] = ' '; }
            TriadString *elem = triad_value_to_string(v.as.lval->items[i]);
            int32_t needed = pos + elem->len + 4;
            if (needed > cap) { cap = needed * 2; out = realloc(out, cap); }
            memcpy(out + pos, elem->data, elem->len);
            pos += elem->len;
            release_str(elem);
        }
        out[pos++] = ']';
        out[pos] = '\0';
        TriadString *s = triad_str_new(out);
        free(out);
        return s;
    }
    default: {
        snprintf(buf, sizeof(buf), "<%s>", tag_name(v.tag));
        return triad_str_new(buf);
    }
    }
}
