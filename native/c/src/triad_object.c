/*
 * TriadLang Native Runtime — Closure + Object
 */
#include "triad_rt.h"
#include <stdlib.h>
#include <string.h>

/* ── Closure ── */

TriadClosure *triad_closure_new(TriadNativeFn fn, int32_t ncaptured) {
    TriadClosure *c = malloc(sizeof(TriadClosure));
    c->refcount = 1;
    c->fn = fn;
    c->ncaptured = ncaptured;
    c->captured = ncaptured > 0 ? calloc(ncaptured, sizeof(TriadValue)) : NULL;
    c->user_data = NULL;
    return c;
}

void triad_closure_free(TriadClosure *c) {
    if (!c) return;
    free(c->captured);
    free(c);
}

TriadValue triad_closure_call(TriadClosure *c, int32_t nargs, TriadValue *args) {
    if (!c || !c->fn) return TRIAD_NONE_VAL;
    int32_t total = nargs + c->ncaptured;
    if (total == 0) return c->fn(0, NULL);
    TriadValue *combined = malloc(total * sizeof(TriadValue));
    if (nargs > 0) memcpy(combined, args, nargs * sizeof(TriadValue));
    if (c->ncaptured > 0) memcpy(combined + nargs, c->captured, c->ncaptured * sizeof(TriadValue));
    TriadValue r = c->fn(total, combined);
    free(combined);
    return r;
}

/* ── Object ── */

TriadObject *triad_object_new(const char *type_name) {
    TriadObject *o = malloc(sizeof(TriadObject));
    o->refcount = 1;
    o->type_name = triad_str_new(type_name);
    o->fields = triad_dict_new();
    return o;
}

void triad_object_free(TriadObject *o) {
    if (!o) return;
    if (o->type_name) triad_str_free(o->type_name);
    if (o->fields) triad_dict_free(o->fields);
    free(o);
}

TriadValue triad_object_get(TriadObject *o, TriadString *key) {
    if (!o || !o->fields) return TRIAD_NONE_VAL;
    return triad_dict_get(o->fields, key);
}

void triad_object_set(TriadObject *o, TriadString *key, TriadValue val) {
    if (!o || !o->fields) return;
    triad_dict_set(o->fields, key, val);
}
