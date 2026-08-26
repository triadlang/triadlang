#include "triad_kernel.h"
#include "triad_mm.h"
#include "triad_paging.h"

static uint8_t *heap_base = NULL;
static uint64_t heap_total = 0;
static uint64_t heap_used = 0;
static uint8_t *heap_ptr = NULL;

typedef struct MmBlock {
    uint32_t size;
    uint32_t free_flag;
    struct MmBlock *next;
} MmBlock;

static MmBlock *free_list = NULL;
static int heap_guard_enabled = 0;

void triad_mm_init(uint64_t mem_size) {
    heap_total = mem_size;
    heap_base = (uint8_t *)0x1000000;
    heap_ptr = heap_base;
    heap_used = 0;
    free_list = NULL;
    /* Guard pages need the paging machinery (bitmap + page tables).
     * Enabled by triad_mm_enable_guard_pages() after triad_paging_init. */
    heap_guard_enabled = 0;
}

void triad_mm_enable_guard_pages(int enable) {
    heap_guard_enabled = enable;
}

static void *alloc_guard_page(void) {
    if (!heap_guard_enabled) return NULL;
    void *page = triad_paging_alloc_page();
    if (!page) return NULL;
    uint64_t virt = (uint64_t)page;
    triad_paging_unmap(virt);
    return page;
}

void *triad_mm_alloc(size_t n) {
    if (n == 0) return NULL;
    n = (n + 15) & ~15;

    MmBlock *prev = NULL;
    MmBlock *blk = free_list;
    while (blk) {
        if (blk->free_flag && blk->size >= n + sizeof(MmBlock)) {
            if (blk->size >= n + sizeof(MmBlock) + 32) {
                MmBlock *split = (MmBlock *)((uint8_t *)blk + sizeof(MmBlock) + n);
                split->size = blk->size - n - sizeof(MmBlock);
                split->free_flag = 1;
                split->next = blk->next;
                blk->size = (uint32_t)n;
                blk->next = split;
            }
            blk->free_flag = 0;
            return (void *)((uint8_t *)blk + sizeof(MmBlock));
        }
        prev = blk;
        blk = blk->next;
    }

    size_t total_needed = n + sizeof(MmBlock);
    if (heap_guard_enabled) {
        total_needed += PAGE_SIZE;
    }

    if (heap_used + total_needed > heap_total) return NULL;

    if (heap_guard_enabled) {
        alloc_guard_page();
    }

    MmBlock *newb = (MmBlock *)heap_ptr;
    heap_ptr += n + sizeof(MmBlock);
    heap_used += n + sizeof(MmBlock);
    newb->size = (uint32_t)n;
    newb->free_flag = 0;
    newb->next = NULL;

    if (prev) {
        prev->next = newb;
    } else {
        free_list = newb;
    }

    return (void *)((uint8_t *)newb + sizeof(MmBlock));
}

void *triad_mm_alloc_aligned(size_t n, size_t align) {
    if (n == 0) return NULL;
    n = (n + (align - 1)) & ~(align - 1);

    uintptr_t p = (uintptr_t)heap_ptr;
    p = (p + align - 1) & ~(align - 1);
    size_t pad = (size_t)(p - (uintptr_t)heap_ptr);
    size_t total = pad + n;

    if (heap_guard_enabled) {
        total += PAGE_SIZE;
    }

    if (heap_used + total > heap_total) return NULL;

    if (heap_guard_enabled) {
        alloc_guard_page();
    }

    heap_ptr = (uint8_t *)(p + n);
    heap_used += total;
    return (void *)p;
}

void *triad_mm_realloc(void *p, size_t n) {
    if (p == NULL) return triad_mm_alloc(n);
    if (n == 0) {
        triad_mm_free(p);
        return NULL;
    }

    MmBlock *blk = (MmBlock *)((uint8_t *)p - sizeof(MmBlock));
    uint32_t old_size = blk->size;
    size_t new_size = (n + 15) & ~15;

    if (new_size <= old_size) {
        blk->size = (uint32_t)new_size;
        return p;
    }

    void *q = triad_mm_alloc(n);
    if (q == NULL) return NULL;

    uint8_t *src = (uint8_t *)p;
    uint8_t *dst = (uint8_t *)q;
    for (uint32_t i = 0; i < old_size; i++) {
        dst[i] = src[i];
    }

    triad_mm_free(p);
    return q;
}

void triad_mm_free(void *p) {
    if (!p) return;
    MmBlock *blk = (MmBlock *)((uint8_t *)p - sizeof(MmBlock));
    blk->free_flag = 1;

    MmBlock *prev = NULL;
    MmBlock *cur = free_list;
    while (cur && cur != blk) {
        prev = cur;
        cur = cur->next;
    }
    if (!cur) return;

    if (cur->next && cur->next->free_flag) {
        cur->size += cur->next->size + sizeof(MmBlock);
        cur->next = cur->next->next;
    }
    if (prev && prev->free_flag) {
        prev->size += cur->size + sizeof(MmBlock);
        prev->next = cur->next;
    }
}

size_t triad_mm_used(void) {
    return (size_t)heap_used;
}

size_t triad_mm_total(void) {
    return (size_t)heap_total;
}

void triad_mm_stats(uint64_t *total, uint64_t *used, uint64_t *free_) {
    *total = heap_total;
    *used = heap_used;
    *free_ = heap_total - heap_used;
}

void triad_mm_mmap_init(TriadMmapEntry *entries, int count) {
    (void)entries;
    (void)count;
}
