#include "triad_aslr.h"
#include "triad_paging.h"
#include "triad_kernel.h"
#include "triad_mm.h"
#include <stdint.h>
#include <stddef.h>

static uint64_t aslr_entropy_pool[4] = {0};
static int aslr_initialized = 0;

static uint64_t xorshift128plus(uint64_t *s) {
    uint64_t s1 = s[0];
    uint64_t s0 = s[1];
    s[0] = s0;
    s1 ^= s1 << 23;
    s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return s[1] + s0;
}

static uint64_t read_tsc(void) {
    uint64_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return (hi << 32) | lo;
}

static int cpu_has_rdrand(void) {
    static int cached = -1;
    if (cached < 0) {
        uint32_t eax, ebx, ecx, edx;
        __asm__ volatile ("cpuid"
                          : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                          : "a"(1), "c"(0));
        cached = (int)((ecx >> 30) & 1);
    }
    return cached;
}

static int cpu_has_rdseed(void) {
    static int cached = -1;
    if (cached < 0) {
        uint32_t eax, ebx, ecx, edx;
        __asm__ volatile ("cpuid"
                          : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                          : "a"(0), "c"(0));
        if (eax < 7) { cached = 0; return cached; }
        __asm__ volatile ("cpuid"
                          : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                          : "a"(7), "c"(0));
        cached = (int)((ebx >> 18) & 1);
    }
    return cached;
}

static int rdrand64(uint64_t *val) {
    if (!cpu_has_rdrand()) return 0;
    int retries = 10;
    uint64_t result;
    while (retries-- > 0) {
        unsigned char ok = 0;
        __asm__ volatile ("rdrand %0; setc %1" : "=r"(result), "=rm"(ok) :: "cc");
        if (ok) {
            *val = result;
            return 1;
        }
    }
    return 0;
}

static int rdseed64(uint64_t *val) {
    if (!cpu_has_rdseed()) return 0;
    int retries = 10;
    uint64_t result;
    while (retries-- > 0) {
        unsigned char ok = 0;
        __asm__ volatile ("rdseed %0; setc %1" : "=r"(result), "=rm"(ok) :: "cc");
        if (ok) {
            *val = result;
            return 1;
        }
    }
    return 0;
}

static uint64_t get_random_entropy(void) {
    uint64_t entropy = 0;

    entropy ^= read_tsc();
    entropy ^= read_tsc() << 17;
    entropy ^= read_tsc() >> 23;

    entropy ^= (uint64_t)&entropy;

    uint64_t rdseed_val;
    if (rdseed64(&rdseed_val)) {
        entropy ^= rdseed_val;
    }

    uint64_t rdrand_val;
    if (rdrand64(&rdrand_val)) {
        entropy ^= rdrand_val;
    }

    entropy ^= xorshift128plus(aslr_entropy_pool);

    return entropy;
}

void triad_aslr_init(void) {
    aslr_entropy_pool[0] = read_tsc();
    aslr_entropy_pool[1] = (uint64_t)&aslr_entropy_pool ^ 0xDEADBEEFCAFEBABEULL;
    aslr_entropy_pool[2] = (uint64_t)&triad_aslr_init ^ 0x123456789ABCDEFULL;
    aslr_entropy_pool[3] = read_tsc() ^ (read_tsc() << 32);

    for (int i = 0; i < 32; i++) {
        aslr_entropy_pool[0] ^= xorshift128plus(aslr_entropy_pool);
        aslr_entropy_pool[1] ^= get_random_entropy();
    }

    aslr_entropy_pool[2] ^= xorshift128plus(aslr_entropy_pool);
    aslr_entropy_pool[3] ^= xorshift128plus(aslr_entropy_pool);

    aslr_initialized = 1;
}

uint64_t triad_aslr_random_offset(int bits) {
    if (bits <= 0) return 0;
    if (bits > 32) bits = 32;

    uint64_t val = xorshift128plus(aslr_entropy_pool);
    val ^= xorshift128plus(aslr_entropy_pool);
    val ^= get_random_entropy();

    uint64_t mask = (1ULL << bits) - 1;
    return (val & mask) * ASLR_ALIGN;
}

uint64_t triad_aslr_random_in_range(uint64_t min, uint64_t max) {
    if (min >= max) return min;

    uint64_t range = max - min;
    uint64_t val = xorshift128plus(aslr_entropy_pool);
    val ^= xorshift128plus(aslr_entropy_pool);

    uint64_t result = min + (val % range);
    result &= ~(ASLR_ALIGN - 1);

    return result;
}

AslrLayout triad_aslr_layout(void) {
    AslrLayout layout;
    layout.entropy = xorshift128plus(aslr_entropy_pool);
    layout.entropy ^= get_random_entropy();

    uint64_t user_space_start = USER_SPACE_BASE + (4 * 1024 * 1024);

    layout.stack_base = USER_SPACE_END - (2 * 1024 * 1024);
    layout.stack_base -= triad_aslr_random_offset(ASLR_STACK_BITS);
    layout.stack_base &= ~(ASLR_ALIGN - 1);

    layout.heap_base = user_space_start;
    layout.heap_base += triad_aslr_random_offset(ASLR_HEAP_BITS);
    layout.heap_base &= ~(ASLR_ALIGN - 1);

    layout.mmap_base = layout.heap_base + (8 * 1024 * 1024);
    layout.mmap_base += triad_aslr_random_offset(ASLR_MMAP_BITS);
    layout.mmap_base &= ~(ASLR_ALIGN - 1);

    layout.brk_base = layout.mmap_base + (64 * 1024 * 1024);
    layout.brk_base += triad_aslr_random_offset(ASLR_BRK_BITS);
    layout.brk_base &= ~(ASLR_ALIGN - 1);

    layout.user_base = user_space_start;
    layout.user_end = layout.stack_base - (8 * 1024 * 1024);

    return layout;
}

AslrLayout triad_aslr_layout_for_process(int pid) {
    AslrLayout layout = triad_aslr_layout();

    layout.entropy ^= ((uint64_t)pid << 32);
    layout.entropy ^= xorshift128plus(aslr_entropy_pool);

    layout.stack_base ^= (uint64_t)pid * ASLR_ALIGN;
    layout.heap_base ^= ((uint64_t)pid << 4) * ASLR_ALIGN;
    layout.mmap_base ^= ((uint64_t)pid << 8) * ASLR_ALIGN;

    layout.stack_base &= ~(ASLR_ALIGN - 1);
    layout.heap_base &= ~(ASLR_ALIGN - 1);
    layout.mmap_base &= ~(ASLR_ALIGN - 1);
    layout.brk_base &= ~(ASLR_ALIGN - 1);

    return layout;
}

uint64_t triad_aslr_stack_top(void) {
    AslrLayout layout = triad_aslr_layout();
    return layout.stack_base;
}

uint64_t triad_aslr_heap_base(void) {
    AslrLayout layout = triad_aslr_layout();
    return layout.heap_base;
}

uint64_t triad_aslr_mmap_base(void) {
    AslrLayout layout = triad_aslr_layout();
    return layout.mmap_base;
}

void triad_aslr_reseed(void) {
    aslr_entropy_pool[0] ^= get_random_entropy();
    aslr_entropy_pool[1] ^= xorshift128plus(aslr_entropy_pool);
    aslr_entropy_pool[2] ^= read_tsc();
    aslr_entropy_pool[3] ^= (uint64_t)&aslr_entropy_pool ^ read_tsc();
}

int triad_aslr_check_addr(uint64_t addr, int is_write) {
    (void)is_write;

    if (addr < USER_SPACE_BASE) {
        return 0;
    }

    if (addr > USER_SPACE_END) {
        return 0;
    }

    if (addr >= 0xFFFF800000000000ULL && addr < 0xFFFFFFFF80000000ULL) {
        return 0;
    }

    return 1;
}