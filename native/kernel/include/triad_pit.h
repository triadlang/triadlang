#ifndef TRIAD_PIT_H
#define TRIAD_PIT_H

#include <stdint.h>

#define PIT_FREQUENCY 1193182
#define PIT_CHANNEL0  0x40
#define PIT_COMMAND   0x43

void triad_pit_init(uint32_t hz);
void triad_pit_set_frequency(uint32_t hz);

#endif