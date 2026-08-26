#ifndef TRIAD_PAGING_H
#define TRIAD_PAGING_H

#include <stdint.h>
#include <stddef.h>

#define PAGE_SIZE    4096ULL
#define PAGE_MASK    (PAGE_SIZE - 1)

#define PTE_PRESENT   (1ULL << 0)
#define PTE_WRITABLE  (1ULL << 1)
#define PTE_USER      (1ULL << 2)
#define PTE_WRITETHROUGH (1ULL << 3)
#define PTE_CACHE_DISABLE (1ULL << 4)
#define PTE_ACCESSED  (1ULL << 5)
#define PTE_DIRTY     (1ULL << 6)
#define PTE_HUGE      (1ULL << 7)
#define PTE_GLOBAL    (1ULL << 8)
#define PTE_NX        (1ULL << 63)

#define PTE_KERNEL    (PTE_PRESENT | PTE_WRITABLE)
#define PTE_KERNEL_RO (PTE_PRESENT)
#define PTE_USER_RW   (PTE_PRESENT | PTE_WRITABLE | PTE_USER)
#define PTE_USER_RO   (PTE_PRESENT | PTE_USER)
#define PTE_USER_EXEC (PTE_PRESENT | PTE_USER)
#define PTE_USER_RX   (PTE_PRESENT | PTE_USER)
#define PTE_USER_RWX  (PTE_PRESENT | PTE_WRITABLE | PTE_USER)

typedef uint64_t PTE;
typedef uint64_t PDE;
typedef uint64_t PDPE;
typedef uint64_t PML4E;

typedef struct {
    PML4E *pml4;
    uint64_t phys_base;
    uint64_t virt_base;
    uint64_t code_start;
    uint64_t code_end;
    uint64_t data_start;
    uint64_t data_end;
    uint64_t stack_base;
    uint64_t stack_top;
    uint64_t heap_base;
    uint64_t heap_end;
    int    is_kernel;
} AddressSpace;

void triad_paging_init(uint64_t mem_size);
void *triad_paging_alloc_page(void);
int  triad_paging_map(uint64_t virt, uint64_t phys, uint32_t flags);
int  triad_paging_unmap(uint64_t virt);
int  triad_paging_set_flags(uint64_t virt, uint64_t flags);
uint64_t triad_paging_phys(uint64_t virt);
void triad_paging_load_cr3(void);
uint64_t triad_paging_free_pages(void);
uint64_t triad_paging_used_pages(void);

void *triad_paging_kmalloc(size_t n);
void triad_paging_kfree(void *p);

AddressSpace *triad_as_create(int is_kernel);
void triad_as_destroy(AddressSpace *as);
int triad_as_map_kernel(AddressSpace *as);
int triad_as_map_user(AddressSpace *as, uint64_t code_start, uint64_t code_size,
                      uint64_t data_start, uint64_t data_size,
                      uint64_t stack_top, uint64_t stack_size);
void triad_as_load(AddressSpace *as);
void triad_as_set_nx(AddressSpace *as, uint64_t start, uint64_t end);
void triad_as_set_ro(AddressSpace *as, uint64_t start, uint64_t end);
void triad_as_set_user(AddressSpace *as, uint64_t start, uint64_t end);
void triad_as_set_kernel_ro(AddressSpace *as, uint64_t start, uint64_t end);

int triad_kpti_init(void);
void triad_kpti_switch_to_kernel(void);
void triad_kpti_switch_to_user(void);
PML4E *triad_kpti_kernel_pml4(void);
PML4E *triad_kpti_user_pml4(void);

#endif
