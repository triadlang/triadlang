#ifndef TRIAD_PIC_H
#define TRIAD_PIC_H

#include <stdint.h>

void triad_pic_init(void);
void triad_pic_eoi(int irq);
void triad_pic_mask(int irq);
void triad_pic_unmask(int irq);

#define PIC1_CMD   0x20
#define PIC1_DATA  0x21
#define PIC2_CMD   0xA0
#define PIC2_DATA  0xA1
#define PIC_EOI    0x20

#define ICW1_ICW4  0x01
#define ICW1_INIT  0x10
#define ICW4_8086  0x01

#endif