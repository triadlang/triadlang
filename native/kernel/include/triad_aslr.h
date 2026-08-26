#ifndef TRIAD_ASLR_H
#define TRIAD_ASLR_H

#include <stdint.h>
#include <stddef.h>

#define ASLR_STACK_BITS  16
#define ASLR_HEAP_BITS   14
#define ASLR_MMAP_BITS   20
#define ASLR_BRK_BITS    12

#define ASLR_ALIGN PAGE_SIZE

typedef struct {
    uint64_t stack_base;
    uint64_t heap_base;
    uint64_t mmap_base;
    uint64_t brk_base;
    uint64_t user_base;
    uint64_t user_end;
    uint64_t entropy;
} AslrLayout;

#define USER_SPACE_BASE  0x00400000ULL
#define USER_SPACE_END   0x7FFFFFFFFFFFULL
#define USER_SPACE_SIZE  (USER_SPACE_END - USER_SPACE_BASE)

void triad_aslr_init(void);
AslrLayout triad_aslr_layout(void);
AslrLayout triad_aslr_layout_for_process(int pid);

uint64_t triad_aslr_stack_top(void);
uint64_t triad_aslr_heap_base(void);
uint64_t triad_aslr_mmap_base(void);

uint64_t triad_aslr_random_offset(int bits);
uint64_t triad_aslr_random_in_range(uint64_t min, uint64_t max);
void triad_aslr_reseed(void);

int triad_aslr_check_addr(uint64_t addr, int is_write);

#endif