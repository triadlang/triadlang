#include "triad_paging.h"
#include "triad_kernel.h"
#include "triad_mm.h"
#include "triad_aslr.h"
#include "triad_serial.h"
#include <string.h>

static uint8_t *page_bitmap = NULL;
static uint64_t total_pages = 0;
static uint64_t used_pages = 0;
static uint64_t next_free_page = 0;

static PML4E *kernel_pml4 = NULL;
static AddressSpace *kernel_as = NULL;
static AddressSpace *current_as = NULL;

static PML4E *kpti_kernel_pml4 = NULL;
static int kpti_enabled = 0;

#define KERNEL_BASE 0xFFFFFFFF80000000ULL
#define KERNEL_PHYS_BASE 0x100000ULL
#define KERNEL_SIZE (16 * 1024 * 1024ULL)

static void *bitmap_alloc_page(void) {
    for (uint64_t i = next_free_page; i < total_pages; i++) {
        uint64_t byte = i / 8;
        uint8_t bit = (uint8_t)(1 << (i % 8));
        if (!(page_bitmap[byte] & bit)) {
            page_bitmap[byte] |= bit;
            used_pages++;
            next_free_page = i + 1;
            return (void *)(0x1000000ULL + i * PAGE_SIZE);
        }
    }
    for (uint64_t i = 0; i < next_free_page; i++) {
        uint64_t byte = i / 8;
        uint8_t bit = (uint8_t)(1 << (i % 8));
        if (!(page_bitmap[byte] & bit)) {
            page_bitmap[byte] |= bit;
            used_pages++;
            next_free_page = i + 1;
            return (void *)(0x1000000ULL + i * PAGE_SIZE);
        }
    }
    return NULL;
}

static void bitmap_free_page(void *page) {
    uint64_t addr = (uint64_t)page;
    if (!page_bitmap) return;
    if (addr < 0x1000000ULL) return;
    uint64_t idx = (addr - 0x1000000ULL) / PAGE_SIZE;
    if (idx >= total_pages) return;
    uint64_t byte = idx / 8;
    uint8_t bit = (uint8_t)(1 << (idx % 8));
    page_bitmap[byte] &= ~bit;
    used_pages--;
    if (idx < next_free_page) next_free_page = idx;
}

void triad_paging_init(uint64_t mem_size) {
    total_pages = mem_size / PAGE_SIZE;
    uint64_t bitmap_size = (total_pages + 7) / 8;
    page_bitmap = (uint8_t *)triad_mm_alloc(bitmap_size);
    memset(page_bitmap, 0, bitmap_size);
    used_pages = 0;
    next_free_page = 0;

    kernel_pml4 = (PML4E *)triad_mm_alloc_aligned(PAGE_SIZE, PAGE_SIZE);
    memset(kernel_pml4, 0, PAGE_SIZE);

    uint64_t kernel_pages = (0x1000000ULL - 0x100000ULL) / PAGE_SIZE;
    for (uint64_t i = 0; i < kernel_pages && i < total_pages; i++) {
        uint64_t byte = i / 8;
        uint8_t bit = (uint8_t)(1 << (i % 8));
        page_bitmap[byte] |= bit;
    }
    used_pages = kernel_pages;
    next_free_page = kernel_pages;

    for (uint64_t addr = 0x100000ULL; addr < 0x1000000ULL; addr += PAGE_SIZE) {
        triad_paging_map(addr, addr, PTE_KERNEL);
    }

    for (uint64_t addr = 0x1000000ULL; addr < 0x1000000ULL + mem_size; addr += PAGE_SIZE) {
        triad_paging_map(addr, addr, PTE_KERNEL);
    }

    triad_paging_map(0x1000000ULL, 0x1000000ULL, PTE_KERNEL);
    triad_paging_map(0x2000000ULL, 0x2000000ULL, PTE_KERNEL);
    triad_paging_map(0x3000000ULL, 0x3000000ULL, PTE_KERNEL);
    triad_paging_map(0xB8000ULL, 0xB8000ULL, PTE_KERNEL);

    kernel_as = triad_as_create(1);
    current_as = kernel_as;
    triad_paging_load_cr3();

    uint64_t cr0;
    __asm__ volatile ("mov %%cr0, %0" : "=r"(cr0));
    cr0 |= (1ULL << 31);
    __asm__ volatile ("mov %0, %%cr0" : : "r"(cr0) : "memory");
}

void *triad_paging_alloc_page(void) {
    void *p = bitmap_alloc_page();
    if (p == NULL) return NULL;
    uint64_t virt = (uint64_t)p;
    uint64_t phys = (uint64_t)p;
    triad_paging_map(virt, phys, PTE_KERNEL);
    return p;
}

static PTE *walk_page_tables(PML4E *pml4, uint64_t virt, bool create) {
    uint64_t pml4_idx = (virt >> 39) & 0x1FF;
    uint64_t pdp_idx   = (virt >> 30) & 0x1FF;
    uint64_t pd_idx    = (virt >> 21) & 0x1FF;
    uint64_t pt_idx    = (virt >> 12) & 0x1FF;

    PML4E *pml4e = &pml4[pml4_idx];
    if (!(*pml4e & PTE_PRESENT)) {
        if (!create) return NULL;
        PDPE *pdp = (PDPE *)triad_mm_alloc_aligned(PAGE_SIZE, PAGE_SIZE);
        if (!pdp) return NULL;
        memset(pdp, 0, PAGE_SIZE);
        *pml4e = (uint64_t)pdp | PTE_PRESENT | PTE_WRITABLE;
    }
    PDPE *pdp = (PDPE *)(*pml4e & ~0xFFF);
    PDPE *pdpe = &pdp[pdp_idx];
    if (!(*pdpe & PTE_PRESENT)) {
        if (!create) return NULL;
        PDE *pd = (PDE *)triad_mm_alloc_aligned(PAGE_SIZE, PAGE_SIZE);
        if (!pd) return NULL;
        memset(pd, 0, PAGE_SIZE);
        *pdpe = (uint64_t)pd | PTE_PRESENT | PTE_WRITABLE;
    }
    PDE *pd = (PDE *)(*pdpe & ~0xFFF);
    PDE *pde = &pd[pd_idx];
    if (!(*pde & PTE_PRESENT)) {
        if (!create) return NULL;
        PTE *pt = (PTE *)triad_mm_alloc_aligned(PAGE_SIZE, PAGE_SIZE);
        if (!pt) return NULL;
        memset(pt, 0, PAGE_SIZE);
        *pde = (uint64_t)pt | PTE_PRESENT | PTE_WRITABLE;
    }
    PTE *pt = (PTE *)(*pde & ~0xFFF);
    return &pt[pt_idx];
}

int triad_paging_map(uint64_t virt, uint64_t phys, uint32_t flags) {
    PTE *pte = walk_page_tables(kernel_pml4, virt, true);
    if (!pte) return -1;
    *pte = (phys & ~0xFFF) | flags;
    __asm__ volatile ("invlpg (%0)" : : "r"(virt) : "memory");
    return 0;
}

int triad_paging_map_as(PML4E *pml4, uint64_t virt, uint64_t phys, uint32_t flags) {
    PTE *pte = walk_page_tables(pml4, virt, true);
    if (!pte) return -1;
    *pte = (phys & ~0xFFF) | flags;
    return 0;
}

int triad_paging_set_flags(uint64_t virt, uint64_t flags) {
    PTE *pte = walk_page_tables(kernel_pml4, virt, false);
    if (!pte || !(*pte & PTE_PRESENT)) return -1;
    *pte = (*pte & 0x000FFFFFFFFFF000ULL) | flags;
    __asm__ volatile ("invlpg (%0)" : : "r"(virt) : "memory");
    return 0;
}

int triad_paging_unmap(uint64_t virt) {
    PTE *pte = walk_page_tables(kernel_pml4, virt, false);
    if (!pte || !(*pte & PTE_PRESENT)) return -1;
    *pte = 0;
    __asm__ volatile ("invlpg (%0)" : : "r"(virt) : "memory");
    return 0;
}

uint64_t triad_paging_phys(uint64_t virt) {
    PTE *pte = walk_page_tables(kernel_pml4, virt, false);
    if (!pte || !(*pte & PTE_PRESENT)) return 0;
    return (*pte & ~0xFFF) | (virt & PAGE_MASK);
}

void triad_paging_load_cr3(void) {
    __asm__ volatile ("mov %0, %%cr3" : : "r"((uint64_t)kernel_pml4));
}

void triad_paging_load_cr3_as(PML4E *pml4) {
    __asm__ volatile ("mov %0, %%cr3" : : "r"((uint64_t)pml4));
}

uint64_t triad_paging_free_pages(void) {
    return total_pages - used_pages;
}

uint64_t triad_paging_used_pages(void) {
    return used_pages;
}

typedef struct KBlock {
    uint32_t size;
    uint32_t free;
    struct KBlock *next;
} KBlock;

static KBlock *kheap_head = NULL;

void *triad_paging_kmalloc(size_t n) {
    n = (n + 15) & ~15;
    if (kheap_head == NULL) {
        kheap_head = (KBlock *)triad_paging_alloc_page();
        if (!kheap_head) return NULL;
        kheap_head->size = PAGE_SIZE;
        kheap_head->free = PAGE_SIZE - sizeof(KBlock);
        kheap_head->next = NULL;
    }
    KBlock *b = kheap_head;
    while (b) {
        if (b->free >= n + sizeof(KBlock)) {
            KBlock *nb = (KBlock *)((uint8_t *)b + b->size - b->free);
            nb->size = (uint32_t)(n + sizeof(KBlock));
            nb->free = (uint32_t)n;
            b->free -= nb->size;
            nb->next = b->next;
            b->next = nb;
            return (void *)((uint8_t *)nb + sizeof(KBlock));
        }
        b = b->next;
    }
    KBlock *nb = (KBlock *)triad_paging_alloc_page();
    if (!nb) return NULL;
    nb->size = PAGE_SIZE;
    nb->free = (uint32_t)(n + sizeof(KBlock));
    nb->next = kheap_head;
    kheap_head = nb;
    return (void *)((uint8_t *)nb + sizeof(KBlock));
}

void triad_paging_kfree(void *p) {
    (void)p;
}

AddressSpace *triad_as_create(int is_kernel) {
    AddressSpace *as = (AddressSpace *)triad_mm_alloc(sizeof(AddressSpace));
    if (!as) return NULL;
    memset(as, 0, sizeof(AddressSpace));

    PML4E *pml4 = (PML4E *)triad_mm_alloc_aligned(PAGE_SIZE, PAGE_SIZE);
    if (!pml4) {
        triad_mm_free(as);
        return NULL;
    }
    memset(pml4, 0, PAGE_SIZE);
    as->pml4 = pml4;
    as->is_kernel = is_kernel;

    return as;
}

void triad_as_destroy(AddressSpace *as) {
    if (!as) return;
    if (as->pml4) {
        triad_mm_free(as->pml4);
    }
    triad_mm_free(as);
}

int triad_as_map_kernel(AddressSpace *as) {
    if (!as || !kernel_pml4) return -1;

    uint64_t kernel_pml4_idx = (KERNEL_BASE >> 39) & 0x1FF;

    as->pml4[kernel_pml4_idx] = kernel_pml4[kernel_pml4_idx];
    as->pml4[kernel_pml4_idx + 1] = kernel_pml4[kernel_pml4_idx + 1];
    as->pml4[kernel_pml4_idx + 2] = kernel_pml4[kernel_pml4_idx + 2];
    as->pml4[kernel_pml4_idx + 3] = kernel_pml4[kernel_pml4_idx + 3];

    for (int i = 0; i < 256; i++) {
        if (kernel_pml4[i] & PTE_PRESENT) {
            as->pml4[i] = kernel_pml4[i];
        }
    }

    return 0;
}

int triad_as_map_user(AddressSpace *as,
                      uint64_t code_start, uint64_t code_size,
                      uint64_t data_start, uint64_t data_size,
                      uint64_t stack_top, uint64_t stack_size) {
    if (!as) return -1;

    for (uint64_t addr = code_start; addr < code_start + code_size; addr += PAGE_SIZE) {
        void *page = bitmap_alloc_page();
        if (!page) return -1;
        uint64_t phys = (uint64_t)page;
        uint64_t virt = addr;
        if (triad_paging_map_as(as->pml4, virt, phys, PTE_USER_EXEC) < 0) {
            bitmap_free_page(page);
            return -1;
        }
    }

    for (uint64_t addr = data_start; addr < data_start + data_size; addr += PAGE_SIZE) {
        void *page = bitmap_alloc_page();
        if (!page) return -1;
        uint64_t phys = (uint64_t)page;
        uint64_t virt = addr;
        if (triad_paging_map_as(as->pml4, virt, phys, PTE_USER_RW) < 0) {
            bitmap_free_page(page);
            return -1;
        }
    }

    for (uint64_t addr = stack_top - stack_size; addr < stack_top; addr += PAGE_SIZE) {
        void *page = bitmap_alloc_page();
        if (!page) return -1;
        uint64_t phys = (uint64_t)page;
        uint64_t virt = addr;
        if (triad_paging_map_as(as->pml4, virt, phys, PTE_USER_RW) < 0) {
            bitmap_free_page(page);
            return -1;
        }
    }

    as->code_start = code_start;
    as->code_end = code_start + code_size;
    as->data_start = data_start;
    as->data_end = data_start + data_size;
    as->stack_base = stack_top - stack_size;
    as->stack_top = stack_top;
    as->heap_base = data_start + data_size;
    as->heap_end = as->heap_base;

    return 0;
}

void triad_as_load(AddressSpace *as) {
    if (!as) {
        __asm__ volatile ("mov %0, %%cr3" : : "r"((uint64_t)kernel_pml4));
        current_as = kernel_as;
        return;
    }
    __asm__ volatile ("mov %0, %%cr3" : : "r"((uint64_t)as->pml4));
    current_as = as;
}

static PTE *triad_as_get_pte(AddressSpace *as, uint64_t virt) {
    return walk_page_tables(as->pml4, virt, false);
}

void triad_as_set_nx(AddressSpace *as, uint64_t start, uint64_t end) {
    if (!as) return;
    for (uint64_t addr = start; addr < end; addr += PAGE_SIZE) {
        PTE *pte = triad_as_get_pte(as, addr);
        if (pte && (*pte & PTE_PRESENT)) {
            *pte |= PTE_NX;
        }
    }
}

void triad_as_set_ro(AddressSpace *as, uint64_t start, uint64_t end) {
    if (!as) return;
    for (uint64_t addr = start; addr < end; addr += PAGE_SIZE) {
        PTE *pte = triad_as_get_pte(as, addr);
        if (pte && (*pte & PTE_PRESENT)) {
            *pte &= ~PTE_WRITABLE;
        }
    }
}

void triad_as_set_user(AddressSpace *as, uint64_t start, uint64_t end) {
    if (!as) return;
    for (uint64_t addr = start; addr < end; addr += PAGE_SIZE) {
        PTE *pte = triad_as_get_pte(as, addr);
        if (pte && (*pte & PTE_PRESENT)) {
            *pte |= PTE_USER;
        }
    }
}

void triad_as_set_kernel_ro(AddressSpace *as, uint64_t start, uint64_t end) {
    if (!as) return;
    for (uint64_t addr = start; addr < end; addr += PAGE_SIZE) {
        PTE *pte = triad_as_get_pte(as, addr);
        if (pte && (*pte & PTE_PRESENT)) {
            *pte &= ~PTE_USER;
            *pte &= ~PTE_WRITABLE;
            *pte |= PTE_NX;
        }
    }
}

int triad_kpti_init(void) {
    kpti_kernel_pml4 = (PML4E *)triad_mm_alloc_aligned(PAGE_SIZE, PAGE_SIZE);
    if (!kpti_kernel_pml4) return -1;
    memset(kpti_kernel_pml4, 0, PAGE_SIZE);

    for (int i = 0; i < 256; i++) {
        if (kernel_pml4[i] & PTE_PRESENT) {
            kpti_kernel_pml4[i] = kernel_pml4[i];
        }
    }

    uint64_t user_pml4_idx_start = (USER_SPACE_BASE >> 39) & 0x1FF;
    uint64_t user_pml4_idx_end = (USER_SPACE_END >> 39) & 0x1FF;
    for (uint64_t i = user_pml4_idx_start; i <= user_pml4_idx_end && i < 256; i++) {
        kpti_kernel_pml4[i] = 0;
    }

    kpti_enabled = 1;
    return 0;
}

void triad_kpti_switch_to_kernel(void) {
    if (!kpti_enabled) return;
    __asm__ volatile ("mov %0, %%cr3" : : "r"((uint64_t)kpti_kernel_pml4) : "memory");
}

void triad_kpti_switch_to_user(void) {
    if (!kpti_enabled) return;
    if (current_as && current_as->pml4) {
        __asm__ volatile ("mov %0, %%cr3" : : "r"((uint64_t)current_as->pml4) : "memory");
    }
}

PML4E *triad_kpti_kernel_pml4(void) {
    return kpti_kernel_pml4;
}

PML4E *triad_kpti_user_pml4(void) {
    if (current_as) return current_as->pml4;
    return kernel_pml4;
}
