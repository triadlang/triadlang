#include "triad_rt.h"
#include <stdlib.h>
#include <string.h>

static void list_grow(TriadList *l, int32_t needed) {
    if (l->cap >= needed) return;
    int32_t newcap = l->cap < 8 ? 8 : l->cap;
    while (newcap < needed) newcap *= 2;
    l->items = realloc(l->items, sizeof(TriadValue) * newcap);
    l->cap = newcap;
}

TriadList *triad_list_new(void) {
    return triad_list_new_cap(8);
}

TriadList *triad_list_new_cap(int32_t cap) {
    TriadList *l = malloc(sizeof(TriadList));
    l->refcount = 1;
    l->len = 0;
    l->cap = cap < 4 ? 4 : cap;
    l->items = malloc(sizeof(TriadValue) * l->cap);
    return l;
}

void triad_list_free(TriadList *l) {
    if (!l) return;
    free(l->items);
    free(l);
}

void triad_list_push(TriadList *l, TriadValue v) {
    list_grow(l, l->len + 1);
    l->items[l->len++] = v;
    triad_retain(&l->items[l->len - 1]);
}

TriadValue triad_list_get(TriadList *l, int32_t idx) {
    if (!l || idx < 0) return TRIAD_NONE_VAL;
    if (idx >= l->len) return TRIAD_NONE_VAL;
    return l->items[idx];
}

void triad_list_set(TriadList *l, int32_t idx, TriadValue v) {
    if (!l || idx < 0 || idx >= l->len) return;
    triad_release(&l->items[idx]);
    l->items[idx] = v;
    triad_retain(&l->items[idx]);
}

int32_t triad_list_len(TriadList *l) {
    return l ? l->len : 0;
}

TriadList *triad_list_slice(TriadList *l, int32_t start, int32_t end, int32_t step) {
    TriadList *r = triad_list_new();
    if (!l) return r;
    if (start < 0) start += l->len;
    if (end < 0) end += l->len;
    if (start < 0) start = 0;
    if (end > l->len) end = l->len;
    if (step == 0) step = 1;

    if (step > 0) {
        for (int32_t i = start; i < end; i += step)
            triad_list_push(r, triad_list_get(l, i));
    } else {
        if (start >= l->len) start = l->len - 1;
        if (end < -1) end = -1;
        for (int32_t i = start; i > end; i += step)
            triad_list_push(r, triad_list_get(l, i));
    }
    return r;
}

static int cmp_int(const void *a, const void *b) {
    int64_t va = ((TriadValue *)a)->as.ival;
    int64_t vb = ((TriadValue *)b)->as.ival;
    return (va > vb) - (va < vb);
}

static int cmp_float(const void *a, const void *b) {
    double va = ((TriadValue *)a)->as.fval;
    double vb = ((TriadValue *)b)->as.fval;
    return (va > vb) - (va < vb);
}

void triad_list_sort(TriadList *l) {
    if (!l || l->len < 2) return;
    if (l->len > 0 && l->items[0].tag == TRIAD_INT)
        qsort(l->items, l->len, sizeof(TriadValue), cmp_int);
    else if (l->len > 0 && l->items[0].tag == TRIAD_FLOAT)
        qsort(l->items, l->len, sizeof(TriadValue), cmp_float);
}

bool triad_list_contains(TriadList *l, TriadValue v) {
    if (!l) return false;
    for (int32_t i = 0; i < l->len; i++)
        if (triad_value_eq(l->items[i], v)) return true;
    return false;
}

TriadList *triad_list_reversed(TriadList *l) {
    TriadList *r = triad_list_new();
    if (!l) return r;
    for (int32_t i = l->len - 1; i >= 0; i--)
        triad_list_push(r, triad_list_get(l, i));
    return r;
}

int32_t triad_list_index_of(TriadList *l, TriadValue v) {
    if (!l) return -1;
    for (int32_t i = 0; i < l->len; i++)
        if (triad_value_eq(l->items[i], v)) return i;
    return -1;
}

void triad_list_insert(TriadList *l, int32_t idx, TriadValue v) {
    if (!l) return;
    if (idx < 0) idx = 0;
    if (idx > l->len) idx = l->len;
    list_grow(l, l->len + 1);
    memmove(&l->items[idx + 1], &l->items[idx], sizeof(TriadValue) * (l->len - idx));
    l->items[idx] = v;
    triad_retain(&l->items[idx]);
    l->len++;
}

void triad_list_remove_at(TriadList *l, int32_t idx) {
    if (!l || idx < 0 || idx >= l->len) return;
    triad_release(&l->items[idx]);
    memmove(&l->items[idx], &l->items[idx + 1], sizeof(TriadValue) * (l->len - idx - 1));
    l->len--;
}
