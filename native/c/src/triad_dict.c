/*
 * TriadLang Native Runtime — Dict (open-addressing hash map)
 */
#include "triad_rt.h"
#include <stdlib.h>
#include <string.h>

#define DICT_LOAD_FACTOR  0.7
#define DICT_TOMBSTONE    ((TriadString *)(intptr_t)-1)

static bool is_tombstone(TriadString *k) {
    return k == DICT_TOMBSTONE;
}

TriadDict *triad_dict_new(void) {
    TriadDict *d = malloc(sizeof(TriadDict));
    d->refcount = 1;
    d->len = 0;
    d->cap = 16;
    d->tombstones = 0;
    d->entries = calloc(d->cap, sizeof(TriadDictEntry));
    return d;
}

void triad_dict_free(TriadDict *d) {
    if (!d) return;
    free(d->entries);
    free(d);
}

static int32_t dict_probe(TriadDict *d, TriadString *key, int32_t hash, bool *found) {
    int32_t idx = hash % d->cap;
    if (idx < 0) idx += d->cap;
    int32_t first_tomb = -1;
    for (int32_t i = 0; i < d->cap; i++) {
        int32_t pos = (idx + i) % d->cap;
        TriadDictEntry *e = &d->entries[pos];
        if (e->key == NULL) {
            if (found) *found = false;
            return (first_tomb >= 0) ? first_tomb : pos;
        }
        if (is_tombstone(e->key)) {
            if (first_tomb < 0) first_tomb = pos;
            continue;
        }
        if (e->hash == hash && triad_str_eq(e->key, key)) {
            if (found) *found = true;
            return pos;
        }
    }
    if (found) *found = false;
    return first_tomb >= 0 ? first_tomb : -1;
}

static void dict_grow(TriadDict *d) {
    int32_t oldcap = d->cap;
    TriadDictEntry *old = d->entries;
    d->cap = oldcap * 2;
    d->entries = calloc(d->cap, sizeof(TriadDictEntry));
    d->len = 0;
    d->tombstones = 0;
    for (int32_t i = 0; i < oldcap; i++) {
        if (old[i].key && !is_tombstone(old[i].key)) {
            triad_dict_set(d, old[i].key, old[i].value);
        }
    }
    free(old);
}

TriadValue triad_dict_get(TriadDict *d, TriadString *key) {
    if (!d || !key) return TRIAD_NONE_VAL;
    int32_t hash = triad_str_hash(key);
    bool found;
    int32_t pos = dict_probe(d, key, hash, &found);
    if (!found || pos < 0) return TRIAD_NONE_VAL;
    return d->entries[pos].value;
}

void triad_dict_set(TriadDict *d, TriadString *key, TriadValue val) {
    if (!d || !key) return;
    if ((double)(d->len + d->tombstones) / d->cap > DICT_LOAD_FACTOR)
        dict_grow(d);

    int32_t hash = triad_str_hash(key);
    bool found;
    int32_t pos = dict_probe(d, key, hash, &found);

    if (pos < 0) { dict_grow(d); pos = dict_probe(d, key, hash, &found); }

    if (found) {
        triad_release(&d->entries[pos].value);
        d->entries[pos].value = val;
        triad_retain(&d->entries[pos].value);
    } else {
        if (is_tombstone(d->entries[pos].key)) {
            d->tombstones--;
        }
        d->entries[pos].key = key;
        d->entries[pos].hash = hash;
        d->entries[pos].value = val;
        triad_retain(&d->entries[pos].value);
        d->len++;
    }
}

bool triad_dict_has(TriadDict *d, TriadString *key) {
    if (!d || !key) return false;
    int32_t hash = triad_str_hash(key);
    bool found;
    dict_probe(d, key, hash, &found);
    return found;
}

void triad_dict_del(TriadDict *d, TriadString *key) {
    if (!d || !key) return;
    int32_t hash = triad_str_hash(key);
    bool found;
    int32_t pos = dict_probe(d, key, hash, &found);
    if (!found || pos < 0) return;
    triad_str_free(d->entries[pos].key);
    triad_release(&d->entries[pos].value);
    d->entries[pos].key = DICT_TOMBSTONE;
    d->len--;
    d->tombstones++;
}

int32_t triad_dict_len(TriadDict *d) {
    return d ? d->len : 0;
}

TriadList *triad_dict_keys(TriadDict *d) {
    TriadList *l = triad_list_new();
    if (!d) return l;
    for (int32_t i = 0; i < d->cap; i++) {
        if (d->entries[i].key && !is_tombstone(d->entries[i].key))
            triad_list_push(l, (TriadValue){.tag = TRIAD_STRING, .as = {.sval = d->entries[i].key}});
    }
    return l;
}

TriadList *triad_dict_values(TriadDict *d) {
    TriadList *l = triad_list_new();
    if (!d) return l;
    for (int32_t i = 0; i < d->cap; i++) {
        if (d->entries[i].key && !is_tombstone(d->entries[i].key))
            triad_list_push(l, d->entries[i].value);
    }
    return l;
}

TriadList *triad_dict_items(TriadDict *d) {
    TriadList *l = triad_list_new();
    if (!d) return l;
    for (int32_t i = 0; i < d->cap; i++) {
        if (d->entries[i].key && !is_tombstone(d->entries[i].key)) {
            TriadList *pair = triad_list_new();
            triad_list_push(pair, (TriadValue){.tag = TRIAD_STRING, .as = {.sval = d->entries[i].key}});
            triad_list_push(pair, d->entries[i].value);
            triad_list_push(l, (TriadValue){.tag = TRIAD_LIST, .as = {.lval = pair}});
        }
    }
    return l;
}
