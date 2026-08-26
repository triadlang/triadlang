#include "triad_serial.h"

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ volatile ("outb %0, %1" : : "a"(val), "Nd"(port));
}

static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    __asm__ volatile ("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

static int serial_initialized = 0;

void triad_serial_init(void) {
    outb(SERIAL_PORT + 1, 0x00);
    outb(SERIAL_PORT + 3, 0x80);
    outb(SERIAL_PORT + 0, 0x03);
    outb(SERIAL_PORT + 1, 0x00);
    outb(SERIAL_PORT + 3, 0x03);
    outb(SERIAL_PORT + 2, 0xC7);
    outb(SERIAL_PORT + 4, 0x0B);
    serial_initialized = 1;
}

static int serial_transmit_empty(void) {
    return inb(SERIAL_PORT + 5) & 0x20;
}

void triad_serial_putc(char c) {
    if (!serial_initialized) triad_serial_init();
    while (!serial_transmit_empty());
    outb(SERIAL_PORT, (uint8_t)c);
}

void triad_serial_puts(const char *s) {
    while (*s) {
        triad_serial_putc(*s);
        s++;
    }
}

void triad_serial_hex(uint64_t val) {
    char buf[17];
    for (int i = 15; i >= 0; i--) {
        uint8_t nib = (uint8_t)(val >> (i * 4)) & 0xF;
        buf[15 - i] = nib < 10 ? '0' + nib : 'a' + nib - 10;
    }
    buf[16] = 0;
    triad_serial_puts(buf);
}

void triad_serial_dec(uint64_t val) {
    char buf[21];
    int i = 20;
    buf[i] = 0;
    if (val == 0) {
        triad_serial_putc('0');
        return;
    }
    while (val > 0 && i > 0) {
        i--;
        buf[i] = '0' + (char)(val % 10);
        val /= 10;
    }
    triad_serial_puts(&buf[i]);
}

void triad_serial_float(double val) {
    if (val < 0) {
        triad_serial_putc('-');
        val = -val;
    }
    triad_serial_dec((uint64_t)val);
    triad_serial_putc('.');
    val -= (double)(uint64_t)val;
    for (int i = 0; i < 4; i++) {
        val *= 10.0;
        triad_serial_putc('0' + (char)(int)val);
        val -= (double)(int)val;
    }
}

char triad_serial_getc(void) {
    if (!serial_initialized) triad_serial_init();
    while (!(inb(SERIAL_PORT + 5) & 1));
    return (char)inb(SERIAL_PORT);
}

int triad_serial_available(void) {
    return inb(SERIAL_PORT + 5) & 1;
}