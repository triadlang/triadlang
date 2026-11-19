/* triad_arena.c — Bump arena for the native frontend. */
#include "triad_frontend.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

struct TriadArenaBlock {
    TriadArenaBlock *next;
    size_t           cap;
    size_t           used;
    /* data follows in the same allocation */
};

static TriadArenaBlock *new_block(size_t cap) {
    TriadArenaBlock *b = (TriadArenaBlock *)malloc(sizeof(TriadArenaBlock) + cap);
    if (!b) return NULL;
    b->next = NULL;
    b->cap  = cap;
    b->used = 0;
    return b;
}

void triad_arena_init(TriadArena *a, size_t block_size) {
    if (!a) return;
    if (block_size < 4096) block_size = 4096;
    a->head        = NULL;
    a->block_size  = block_size;
    a->total_bytes = 0;
}

static void *block_data(TriadArenaBlock *b) {
    return (void *)(b + 1);
}

void *triad_arena_alloc(TriadArena *a, size_t n) {
    if (!a) return NULL;
    if (n == 0) n = 1;
    /* 16-byte alignment */
    size_t aligned = (n + 15u) & ~(size_t)15u;
    TriadArenaBlock *b = a->head;
    if (!b || b->used + aligned > b->cap) {
        size_t cap = a->block_size;
        if (aligned > cap) cap = aligned + 64;
        TriadArenaBlock *nb = new_block(cap);
        if (!nb) return NULL;
        nb->next = a->head;
        a->head  = nb;
        b        = nb;
    }
    void *p = (char *)block_data(b) + b->used;
    b->used        += aligned;
    a->total_bytes += aligned;
    return p;
}

void *triad_arena_calloc(TriadArena *a, size_t n) {
    void *p = triad_arena_alloc(a, n);
    if (p) memset(p, 0, n);
    return p;
}

char *triad_arena_strdup(TriadArena *a, const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    return triad_arena_strndup(a, s, n);
}

char *triad_arena_strndup(TriadArena *a, const char *s, size_t n) {
    char *out = (char *)triad_arena_alloc(a, n + 1);
    if (!out) return NULL;
    if (n) memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

void triad_arena_free(TriadArena *a) {
    if (!a) return;
    TriadArenaBlock *b = a->head;
    while (b) {
        TriadArenaBlock *nx = b->next;
        free(b);
        b = nx;
    }
    a->head        = NULL;
    a->total_bytes = 0;
}
