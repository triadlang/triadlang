#ifndef TRIAD_MM_H
#define TRIAD_MM_H

#include "triad_kernel.h"

void     triad_mm_init(uint64_t mem_size);
void     triad_mm_enable_guard_pages(int enable);
void    *triad_mm_alloc(size_t n);
void    *triad_mm_alloc_aligned(size_t n, size_t align);
void    *triad_mm_realloc(void *p, size_t n);
void     triad_mm_free(void *p);
size_t   triad_mm_used(void);
size_t   triad_mm_total(void);
void     triad_mm_stats(uint64_t *total, uint64_t *used, uint64_t *free);

typedef struct {
    uint64_t base;
    uint64_t len;
    uint32_t type;
} TriadMmapEntry;

void triad_mm_mmap_init(TriadMmapEntry *entries, int count);

#endif