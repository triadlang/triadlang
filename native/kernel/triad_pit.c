#include "triad_pit.h"

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ volatile ("outb %0, %1" : : "a"(val), "Nd"(port));
}

void triad_pit_init(uint32_t hz) {
    triad_pit_set_frequency(hz);
}

void triad_pit_set_frequency(uint32_t hz) {
    uint32_t divisor = PIT_FREQUENCY / hz;
    if (divisor < 1) divisor = 1;
    if (divisor > 65535) divisor = 65535;
    outb(PIT_COMMAND, 0x36);
    outb(PIT_CHANNEL0, (uint8_t)(divisor & 0xFF));
    outb(PIT_CHANNEL0, (uint8_t)((divisor >> 8) & 0xFF));
}