#ifndef TRIAD_ISR_H
#define TRIAD_ISR_H

#include <stdint.h>

typedef struct {
    uint64_t r15, r14, r13, r12, r11, r10, r9, r8;
    uint64_t rdi, rsi, rbp, rdx, rcx, rbx, rax;
    uint64_t int_no, err_code;
    uint64_t rip, cs, rflags, rsp, ss;
} __attribute__((packed)) TriadRegisters;

extern uint64_t isr_table[256];

TriadRegisters *isr_handler(TriadRegisters *regs);
void irq_handler(TriadRegisters *regs);
void isr_table_init(void);

#define IRQ_TIMER    32
#define IRQ_KEYBOARD 33
#define IRQ_SERIAL   36
#define IRQ_SYSCALL  128

typedef void (*TriadIsrCallback)(TriadRegisters *);

void triad_isr_register(int irq, TriadIsrCallback cb);
void triad_isr_eoi(int irq);

#endif