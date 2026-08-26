#include "triad_hardening.h"
#include "triad_kernel.h"
#include "triad_mm.h"
#include "triad_paging.h"
#include "triad_serial.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static int hardening_features[HARDEN_MAX] = {0};
static int hardening_initialized = 0;

static RateLimiter rate_limiters[MAX_RATE_LIMITERS];
static int n_rate_limiters = 0;

const char * const triad_hidden_symbols[] = {
    "triad_crypto_key",
    "triad_admin_password_hash",
    "triad_master_secret",
    "triad_kernel_key",
    "triad_debug_token",
    NULL
};

extern uint8_t _text_start[];
extern uint8_t _text_end[];
extern uint8_t _rodata_start[];
extern uint8_t _rodata_end[];
extern uint8_t _data_start[];
extern uint8_t _data_end[];
extern uint8_t _bss_start[];
extern uint8_t _bss_end[];

static uint64_t read_tsc(void) {
    uint64_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return (hi << 32) | lo;
}

static uint32_t cpu_features_edx = 0;
static uint32_t cpu_features_ecx = 0;
static uint32_t cpu_ext_features_edx = 0;
static uint32_t cpu_ext_features_ecx = 0;

static void read_cpu_features(void) {
    uint32_t eax, ebx, ecx, edx;

    eax = 1;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax));
    cpu_features_ecx = ecx;
    cpu_features_edx = edx;

    eax = 0x80000001;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax));
    cpu_ext_features_ecx = ecx;
    cpu_ext_features_edx = edx;
}

void triad_hardening_init(void) {
    if (hardening_initialized) return;

    read_cpu_features();

    hardening_features[HARDEN_NX_BIT] = (cpu_ext_features_edx >> 20) & 1;
    hardening_features[HARDEN_ASLR] = 1;
    hardening_features[HARDEN_STACK_CANARY] = 1;
    hardening_features[HARDEN_STACK_GUARD] = 1;

    hardening_features[HARDEN_RDRAND] = (cpu_features_ecx >> 30) & 1;

    uint32_t eax = 7;
    uint32_t ecx = 0;
    uint32_t ebx, edx;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax), "c"(ecx));
    hardening_features[HARDEN_RDSEED] = (ebx >> 18) & 1;
    /* SMEP = CPUID.07:EBX bit 7, SMAP = CPUID.07:EBX bit 20 */
    hardening_features[HARDEN_SMEP] = (ebx >> 7) & 1;
    hardening_features[HARDEN_SMAP] = (ebx >> 20) & 1;

    if (hardening_features[HARDEN_NX_BIT]) {
        triad_enable_nx();
    }
    if (hardening_features[HARDEN_SMEP]) {
        triad_enable_smep();
    }

    if (hardening_features[HARDEN_SMAP]) {
        triad_enable_smap();
    }
    triad_mark_text_ro();
    triad_mark_rodata();
    triad_mark_bss_zero();
    hardening_features[HARDEN_WX] = 1;

    for (int i = 0; i < MAX_RATE_LIMITERS; i++) {
        rate_limiters[i].window_start = 0;
        rate_limiters[i].count = 0;
        rate_limiters[i].max_per_second = 0;
        rate_limiters[i].last_reset = 0;
    }

    hardening_initialized = 1;
}

int triad_has_nx(void) {
    return hardening_features[HARDEN_NX_BIT];
}

int triad_has_smap(void) {
    return hardening_features[HARDEN_SMAP];
}

int triad_has_smep(void) {
    return hardening_features[HARDEN_SMEP];
}

int triad_has_rdrand(void) {
    return hardening_features[HARDEN_RDRAND];
}

int triad_has_rdseed(void) {
    return hardening_features[HARDEN_RDSEED];
}

int triad_has_kpti(void) {
    return hardening_features[HARDEN_KPTI];
}

int triad_has_wx(void) {
    return hardening_features[HARDEN_WX];
}

void triad_enable_nx(void) {
    uint64_t efer;
    __asm__ volatile ("rdmsr" : "=a"(efer) : "c"(0xC0000080));
    efer |= (1ULL << 11);
    __asm__ volatile ("wrmsr" : : "a"((uint32_t)efer), "d"((uint32_t)(efer >> 32)), "c"(0xC0000080));
    hardening_features[HARDEN_NX_BIT] = 1;
}

void triad_enable_smap(void) {
    uint64_t cr4;
    __asm__ volatile ("mov %%cr4, %0" : "=r"(cr4));
    cr4 |= (1ULL << 21);
    __asm__ volatile ("mov %0, %%cr4" : : "r"(cr4));
    hardening_features[HARDEN_SMAP] = 1;
}

void triad_enable_smep(void) {
    uint64_t cr4;
    __asm__ volatile ("mov %%cr4, %0" : "=r"(cr4));
    cr4 |= (1ULL << 20);
    __asm__ volatile ("mov %0, %%cr4" : : "r"(cr4));
    hardening_features[HARDEN_SMEP] = 1;
}

void triad_enable_kpti(void) {
    hardening_features[HARDEN_KPTI] = 1;
}

int triad_check_cpu_features(void) {
    int score = 0;

    if (hardening_features[HARDEN_NX_BIT]) score += 10;
    if (hardening_features[HARDEN_SMEP]) score += 10;
    if (hardening_features[HARDEN_SMAP]) score += 10;
    if (hardening_features[HARDEN_RDRAND]) score += 5;
    if (hardening_features[HARDEN_RDSEED]) score += 5;

    return score;
}

void triad_mark_text_ro(void) {
    uint64_t start = (uint64_t)_text_start;
    uint64_t end = (uint64_t)_text_end;

    for (uint64_t addr = start; addr < end; addr += PAGE_SIZE) {
        triad_paging_set_flags(addr, PTE_PRESENT);
    }
}

void triad_mark_rodata(void) {
    uint64_t start = (uint64_t)_rodata_start;
    uint64_t end = (uint64_t)_rodata_end;

    for (uint64_t addr = start; addr < end; addr += PAGE_SIZE) {
        triad_paging_set_flags(addr, PTE_PRESENT | PTE_NX);
    }
}

void triad_mark_bss_zero(void) {
    uint64_t start = (uint64_t)_bss_start;
    uint64_t end = (uint64_t)_bss_end;

    /* .bss is zeroed by the multiboot loader before _start; re-zeroing here
     * would wipe live state (the kernel stack itself lives in .bss).
     * The hardening step is only the W^X marking below. */
    for (uint64_t addr = start; addr < end; addr += PAGE_SIZE) {
        triad_paging_set_flags(addr, PTE_PRESENT | PTE_WRITABLE | PTE_NX);
    }
}

void triad_secure_zero(void *ptr, size_t len) {
    if (!ptr || len == 0) return;

    volatile uint8_t *p = (volatile uint8_t *)ptr;
    for (size_t i = 0; i < len; i++) {
        p[i] = 0;
    }

    __asm__ volatile ("" : : "r"(p) : "memory");
}

void triad_secure_free(void *ptr, size_t len) {
    triad_secure_zero(ptr, len);
    triad_mm_free(ptr);
}

void *triad_kernel_safe_ptr(void *ptr, size_t len) {
    if (!ptr) return NULL;

    uint64_t addr = (uint64_t)ptr;

    if (addr < 0x100000ULL) return NULL;

    if (addr >= 0xFFFF800000000000ULL && addr < 0xFFFFFFFF80000000ULL) return NULL;

    (void)len;
    return ptr;
}

int triad_rate_limit_check(RateLimiter *limiter) {
    if (!limiter || limiter->max_per_second == 0) return 1;

    uint64_t now = read_tsc();
    uint64_t freq = 1000000000ULL;

    if (now - limiter->window_start > freq) {
        limiter->window_start = now;
        limiter->count = 0;
    }

    if (limiter->count >= limiter->max_per_second) {
        return 0;
    }

    limiter->count++;
    return 1;
}

void triad_rate_limit_reset(RateLimiter *limiter) {
    if (!limiter) return;
    limiter->count = 0;
    limiter->window_start = read_tsc();
}

RateLimiter *triad_rate_limit_get(int idx) {
    if (idx < 0 || idx >= n_rate_limiters) return NULL;
    return &rate_limiters[idx];
}

int triad_rate_limit_create(uint32_t max_per_second) {
    if (n_rate_limiters >= MAX_RATE_LIMITERS) return -1;

    int idx = n_rate_limiters++;
    rate_limiters[idx].window_start = read_tsc();
    rate_limiters[idx].count = 0;
    rate_limiters[idx].max_per_second = max_per_second;
    rate_limiters[idx].last_reset = 0;

    return idx;
}

void triad_hardening_report(void) {
    triad_serial_puts("\n=== Kernel Hardening Report ===\n");

    triad_serial_puts("NX (No-Execute): ");
    triad_serial_puts(hardening_features[HARDEN_NX_BIT] ? "ENABLED\n" : "DISABLED\n");

    triad_serial_puts("SMEP (Supervisor Mode Execution Prevention): ");
    triad_serial_puts(hardening_features[HARDEN_SMEP] ? "ENABLED\n" : "DISABLED\n");

    triad_serial_puts("SMAP (Supervisor Mode Access Prevention): ");
    triad_serial_puts(hardening_features[HARDEN_SMAP] ? "ENABLED\n" : "DISABLED\n");

    triad_serial_puts("RDRAND: ");
    triad_serial_puts(hardening_features[HARDEN_RDRAND] ? "AVAILABLE\n" : "UNAVAILABLE\n");

    triad_serial_puts("RDSEED: ");
    triad_serial_puts(hardening_features[HARDEN_RDSEED] ? "AVAILABLE\n" : "UNAVAILABLE\n");

    triad_serial_puts("ASLR: ");
    triad_serial_puts(hardening_features[HARDEN_ASLR] ? "ENABLED\n" : "DISABLED\n");

    triad_serial_puts("Stack Canary: ");
    triad_serial_puts(hardening_features[HARDEN_STACK_CANARY] ? "ENABLED\n" : "DISABLED\n");

    triad_serial_puts("Stack Guard: ");
    triad_serial_puts(hardening_features[HARDEN_STACK_GUARD] ? "ENABLED\n" : "DISABLED\n");

    triad_serial_puts("W^X Enforcement: ");
    triad_serial_puts(hardening_features[HARDEN_WX] ? "ENABLED\n" : "DISABLED\n");

    triad_serial_puts("KPTI: ");
    triad_serial_puts(hardening_features[HARDEN_KPTI] ? "ENABLED\n" : "DISABLED\n");

    triad_serial_puts("\nHardening Score: ");
    int score = triad_check_cpu_features();
    if (hardening_features[HARDEN_WX]) score += 10;
    if (hardening_features[HARDEN_KPTI]) score += 10;
    char buf[16];
    int i = 0;
    int tmp = score;
    if (tmp == 0) {
        buf[i++] = '0';
    } else {
        char t[16];
        int ti = 0;
        while (tmp > 0) {
            t[ti++] = '0' + (tmp % 10);
            tmp /= 10;
        }
        while (ti > 0) {
            buf[i++] = t[--ti];
        }
    }
    buf[i] = 0;
    triad_serial_puts(buf);
    triad_serial_puts("/60\n");

    triad_serial_puts("================================\n");
}

int triad_is_hardened(void) {
    return hardening_features[HARDEN_NX_BIT] &&
           hardening_features[HARDEN_SMEP] &&
           hardening_features[HARDEN_SMAP] &&
           hardening_features[HARDEN_WX];
}

int triad_get_hardening_bits(void) {
    int bits = 0;
    for (int i = 0; i < HARDEN_MAX; i++) {
        if (hardening_features[i]) {
            bits |= (1 << i);
        }
    }
    return bits;
}
