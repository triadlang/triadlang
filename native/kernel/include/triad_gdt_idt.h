#ifndef TRIAD_GDT_IDT_H
#define TRIAD_GDT_IDT_H

#include <stdint.h>

#define GDT_ENTRIES 7
#define IDT_ENTRIES 256

#define KERNEL_CS 0x08
#define KERNEL_DS 0x10
#define USER_CS   0x18
#define USER_DS   0x20

typedef struct {
    uint16_t limit_low;
    uint16_t base_low;
    uint8_t  base_mid;
    uint8_t  access;
    uint8_t  granularity;
    uint8_t  base_high;
} __attribute__((packed)) GDTEntry;

typedef struct {
    uint16_t limit;
    uint64_t base;
} __attribute__((packed)) GDTPtr;

typedef struct {
    uint16_t offset_low;
    uint16_t selector;
    uint8_t  ist;
    uint8_t  type_attr;
    uint16_t offset_mid;
    uint32_t offset_high;
    uint32_t zero;
} __attribute__((packed)) IDTEntry;

typedef struct {
    uint16_t limit;
    uint64_t base;
} __attribute__((packed)) IDTPtr;

extern GDTEntry gdt[GDT_ENTRIES];

void gdt_init(void);
void idt_init(void);
void idt_set_gate(int irq, uint64_t handler, uint8_t type_attr, uint8_t ist);
void irq_enable(void);
void irq_disable(void);

#endif