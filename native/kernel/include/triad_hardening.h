#ifndef TRIAD_HARDENING_H
#define TRIAD_HARDENING_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    uint64_t window_start;
    uint32_t count;
    uint32_t max_per_second;
    uint64_t last_reset;
} RateLimiter;

#define MAX_RATE_LIMITERS 16

typedef enum {
    HARDEN_NX_BIT = 0,
    HARDEN_ASLR = 1,
    HARDEN_STACK_CANARY = 2,
    HARDEN_STACK_GUARD = 3,
    HARDEN_HEAP_GUARD = 4,
    HARDEN_KPTI = 5,
    HARDEN_SMEP = 6,
    HARDEN_SMAP = 7,
    HARDEN_RDRAND = 8,
    HARDEN_RDSEED = 9,
    HARDEN_WX = 10,
    HARDEN_MAX = 11
} HardeningFeature;

void triad_hardening_init(void);

int triad_has_nx(void);
int triad_has_smap(void);
int triad_has_smep(void);
int triad_has_rdrand(void);
int triad_has_rdseed(void);
int triad_has_kpti(void);
int triad_has_wx(void);

void triad_enable_nx(void);
void triad_enable_smap(void);
void triad_enable_smep(void);
void triad_enable_kpti(void);

int triad_check_cpu_features(void);

void triad_mark_text_ro(void);
void triad_mark_rodata(void);
void triad_mark_bss_zero(void);

void triad_secure_zero(void *ptr, size_t len);
void triad_secure_free(void *ptr, size_t len);

void *triad_kernel_safe_ptr(void *ptr, size_t len);

int triad_rate_limit_check(RateLimiter *limiter);
void triad_rate_limit_reset(RateLimiter *limiter);
RateLimiter *triad_rate_limit_get(int idx);
int triad_rate_limit_create(uint32_t max_per_second);

void triad_hardening_report(void);

int triad_is_hardened(void);
int triad_get_hardening_bits(void);

extern const char * const triad_hidden_symbols[];

static inline void triad_stac(void) {
    __asm__ volatile ("stac" ::: "memory");
}

static inline void triad_clac(void) {
    __asm__ volatile ("clac" ::: "memory");
}

#endif
