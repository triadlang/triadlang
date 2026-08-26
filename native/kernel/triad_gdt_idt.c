#include "triad_gdt_idt.h"
#include "triad_isr.h"
#include <string.h>

GDTEntry gdt[GDT_ENTRIES];
static GDTPtr gdtptr;

static IDTEntry idt[IDT_ENTRIES];
static IDTPtr idtptr;

static void gdt_set(int i, uint32_t base, uint32_t limit, uint8_t access, uint8_t gran) {
    gdt[i].limit_low   = (uint16_t)(limit & 0xFFFF);
    gdt[i].base_low    = (uint16_t)(base & 0xFFFF);
    gdt[i].base_mid    = (uint8_t)((base >> 16) & 0xFF);
    gdt[i].access      = access;
    gdt[i].granularity = (uint8_t)((limit >> 16) & 0x0F);
    gdt[i].granularity |= (uint8_t)(gran & 0xF0);
    gdt[i].base_high   = (uint8_t)((base >> 24) & 0xFF);
}

void gdt_init(void) {
    memset(gdt, 0, sizeof(gdt));
    gdt_set(0, 0, 0, 0, 0);
    gdt_set(1, 0, 0xFFFFF, 0x9A, 0xA0);
    gdt_set(2, 0, 0xFFFFF, 0x92, 0xC0);
    gdt_set(3, 0, 0xFFFFF, 0xFA, 0xA0);
    gdt_set(4, 0, 0xFFFFF, 0xF2, 0xC0);
    gdt_set(5, 0, 0, 0x89, 0x00);
    gdtptr.limit = sizeof(gdt) - 1;
    gdtptr.base  = (uint64_t)&gdt;
    __asm__ volatile ("lgdt %0" : : "m"(gdtptr));
    __asm__ volatile (
        "movw %w0, %%ds\n"
        "movw %w0, %%es\n"
        "movw %w0, %%fs\n"
        "movw %w0, %%gs\n"
        "movw %w0, %%ss\n"
        "pushq %1\n"
        "leaq 1f(%%rip), %%rax\n"
        "pushq %%rax\n"
        "lretq\n"
        "1:\n"
        : : "a"((uint16_t)KERNEL_DS), "ri"((uint16_t)KERNEL_CS)
    );
}

void idt_set_gate(int irq, uint64_t handler, uint8_t type_attr, uint8_t ist) {
    idt[irq].offset_low  = (uint16_t)(handler & 0xFFFF);
    idt[irq].selector     = KERNEL_CS;
    idt[irq].ist          = ist;
    idt[irq].type_attr    = type_attr;
    idt[irq].offset_mid   = (uint16_t)((handler >> 16) & 0xFFFF);
    idt[irq].offset_high  = (uint32_t)((handler >> 32) & 0xFFFFFFFF);
    idt[irq].zero         = 0;
}

void idt_init(void) {
    memset(idt, 0, sizeof(idt));
    idtptr.limit = sizeof(idt) - 1;
    idtptr.base  = (uint64_t)&idt;

    for (int i = 0; i < 32; i++) {
        idt_set_gate(i, (uint64_t)isr_table[i], 0x8E, 0);
    }
    for (int i = 32; i < 48; i++) {
        idt_set_gate(i, (uint64_t)isr_table[i], 0x8E, 0);
    }
    idt_set_gate(0x80, (uint64_t)isr_table[128], 0xEE, 0);
    __asm__ volatile ("lidt %0" : : "m"(idtptr));
}

void irq_enable(void) {
    __asm__ volatile ("sti");
}

void irq_disable(void) {
    __asm__ volatile ("cli");
}