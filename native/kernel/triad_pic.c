#include "triad_pic.h"
#include "triad_isr.h"

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ volatile ("outb %0, %1" : : "a"(val), "Nd"(port));
}

static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    __asm__ volatile ("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

void triad_pic_init(void) {
    outb(PIC1_CMD, ICW1_INIT | ICW1_ICW4);
    outb(PIC2_CMD, ICW1_INIT | ICW1_ICW4);
    outb(PIC1_DATA, 0x20);
    outb(PIC2_DATA, 0x28);
    outb(PIC1_DATA, 0x04);
    outb(PIC2_DATA, 0x02);
    outb(PIC1_DATA, ICW4_8086);
    outb(PIC2_DATA, ICW4_8086);

    uint8_t mask1 = inb(PIC1_DATA) | 0xFF;
    uint8_t mask2 = inb(PIC2_DATA) | 0xFF;
    outb(PIC1_DATA, mask1);
    outb(PIC2_DATA, mask2);
}

void triad_pic_eoi(int irq) {
    if (irq >= 40) {
        outb(PIC2_CMD, PIC_EOI);
    }
    outb(PIC1_CMD, PIC_EOI);
}

void triad_pic_mask(int irq) {
    uint16_t port;
    uint8_t value;
    if (irq < 8) {
        port = PIC1_DATA;
    } else if (irq < 16) {
        port = PIC2_DATA;
        irq -= 8;
    } else {
        return;
    }
    value = inb(port) | (1 << irq);
    outb(port, value);
}

void triad_pic_unmask(int irq) {
    uint16_t port;
    uint8_t value;
    if (irq < 8) {
        port = PIC1_DATA;
    } else if (irq < 16) {
        port = PIC2_DATA;
        irq -= 8;
    } else {
        return;
    }
    value = inb(port) & ~(1 << irq);
    outb(port, value);
}