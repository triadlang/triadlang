#include "triad_stack_protect.h"
#include "triad_kernel.h"
#include "triad_serial.h"
#include <stdint.h>

uint64_t __stack_chk_guard = 0;

static uint64_t rng_state[2] = {0x123456789ABCDEFULL, 0xFEDCBA9876543210ULL};

static uint64_t xorshift128plus(void) {
    uint64_t s1 = rng_state[0];
    uint64_t s0 = rng_state[1];
    rng_state[0] = s0;
    s1 ^= s1 << 23;
    rng_state[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return rng_state[1] + s0;
}

static uint64_t read_tsc(void) {
    uint64_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return (hi << 32) | lo;
}

static int rdrand_available(void) {
    uint32_t eax, ebx, ecx, edx;
    eax = 1;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax));
    return (ecx & (1 << 30)) != 0;
}

static int rdrand64(uint64_t *val) {
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

static int rdseed_available(void) {
    uint32_t eax, ebx, ecx, edx;
    eax = 7;
    ecx = 0;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax), "c"(ecx));
    return (ebx & (1 << 18)) != 0;
}

static int rdseed64(uint64_t *val) {
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

void triad_stack_canary_init(void) {
    uint64_t seed = read_tsc();
    rng_state[0] ^= seed;
    rng_state[1] ^= seed >> 32;
    rng_state[0] ^= read_tsc() ^ (read_tsc() << 17);

    if (rdseed_available()) {
        uint64_t rdseed_val;
        if (rdseed64(&rdseed_val)) {
            __stack_chk_guard = rdseed_val;
            return;
        }
    }

    if (rdrand_available()) {
        uint64_t rdrand_val;
        if (rdrand64(&rdrand_val)) {
            __stack_chk_guard = rdrand_val;
            return;
        }
    }

    __stack_chk_guard = xorshift128plus();
    __stack_chk_guard ^= read_tsc();
    __stack_chk_guard ^= ((uint64_t)&__stack_chk_guard << 32);
    __stack_chk_guard |= (1ULL << 32);
}

uint64_t triad_stack_random_canary(void) {
    uint64_t canary = 0;

    if (rdrand_available()) {
        if (rdrand64(&canary)) {
            return canary;
        }
    }

    canary = xorshift128plus();
    canary ^= read_tsc();
    return canary;
}

uint64_t __stack_chk_setup(void) {
    return __stack_chk_guard;
}

void __stack_chk_fail(void) {
    triad_stack_panic("stack smashing detected");
}

int triad_stack_check_frame(StackFrame *frame) {
    if (!frame) return 0;
    return frame->canary == __stack_chk_guard;
}

void triad_stack_panic(const char *msg) {
    triad_serial_puts("\n!!! KERNEL PANIC !!!\n");
    triad_serial_puts("Stack protection: ");
    triad_serial_puts(msg ? msg : "(no message)");
    triad_serial_puts("\n");

    triad_serial_puts("Canary value: 0x");
    for (int i = 60; i >= 0; i -= 4) {
        uint8_t nibble = (uint8_t)((__stack_chk_guard >> i) & 0xF);
        triad_serial_putc(nibble < 10 ? '0' + nibble : 'A' + nibble - 10);
    }
    triad_serial_puts("\n");

    uint64_t rbp;
    __asm__ volatile ("mov %%rbp, %0" : "=r"(rbp));
    triad_serial_puts("RBP: 0x");
    for (int i = 60; i >= 0; i -= 4) {
        uint8_t nibble = (uint8_t)((rbp >> i) & 0xF);
        triad_serial_putc(nibble < 10 ? '0' + nibble : 'A' + nibble - 10);
    }
    triad_serial_puts("\n");

    triad_serial_puts("Stack trace:\n");
    int depth = 0;
    uint64_t *frame_ptr = (uint64_t *)rbp;
    while (frame_ptr && depth < 16) {
        uint64_t rip = frame_ptr[1];
        triad_serial_puts("  [");
        triad_serial_putc('0' + depth);
        triad_serial_puts("] 0x");
        for (int i = 60; i >= 0; i -= 4) {
            uint8_t nibble = (uint8_t)((rip >> i) & 0xF);
            triad_serial_putc(nibble < 10 ? '0' + nibble : 'A' + nibble - 10);
        }
        triad_serial_puts("\n");
        frame_ptr = (uint64_t *)frame_ptr[0];
        depth++;
    }

    __asm__ volatile ("cli; hlt");
    __builtin_unreachable();
}