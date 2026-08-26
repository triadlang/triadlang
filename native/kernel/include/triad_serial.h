#ifndef TRIAD_SERIAL_H
#define TRIAD_SERIAL_H

#include <stdint.h>

#define SERIAL_PORT 0x3F8

void triad_serial_init(void);
void triad_serial_putc(char c);
void triad_serial_puts(const char *s);
void triad_serial_hex(uint64_t val);
void triad_serial_dec(uint64_t val);
void triad_serial_float(double val);
char triad_serial_getc(void);
int triad_serial_available(void);

#endif