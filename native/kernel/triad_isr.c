#include "triad_isr.h"
#include "triad_gdt_idt.h"
#include "triad_pic.h"
#include "triad_serial.h"
#include "triad_sched.h"
#include "triad_syscall.h"
#include "triad_vfs.h"
#include "triad_mm.h"
#include "triad_kernel.h"
#include "triad_equilibrium.h"
#include "triad_paging.h"
#include "triad_hardening.h"
#include <string.h>

uint64_t isr_table[256];

static TriadIsrCallback callbacks[256];

static const char *exc_names[32] = {
    "divide by zero", "debug", "nmi", "breakpoint", "overflow",
    "bound range exceeded", "invalid opcode", "device not available",
    "double fault", "coprocessor segment overrun", "invalid tss",
    "segment not present", "stack segment fault", "general protection fault",
    "page fault", "reserved", "x87 fpu error", "alignment check",
    "machine check", "simd fpu error", "virtualization error", "reserved",
    "reserved", "reserved", "reserved", "reserved", "reserved",
    "reserved", "reserved", "reserved", "security", "reserved",
};

void triad_isr_register(int irq, TriadIsrCallback cb) {
    if (irq >= 0 && irq < 256) {
        callbacks[irq] = cb;
    }
}

void triad_isr_eoi(int irq) {
    triad_pic_eoi(irq);
}

TriadRegisters *isr_handler(TriadRegisters *regs) {
    uint64_t int_no = regs->int_no;

    if (int_no < 32) {
        const char *name = exc_names[int_no];
        triad_serial_puts("\n!!! kernel panic: ");
        triad_serial_puts(name);
        triad_serial_puts("\n");
        while (1) {
            __asm__ volatile ("hlt");
        }
    }

    if (int_no == IRQ_TIMER) {
        triad_sched_tick();
        triad_eq_update(triad_sched_get_ticks());

        if (triad_sched_needs_switch()) {
            TriadThread *current = triad_sched_current();
            if (current) {
                current->ctx.rsp = (uint64_t)regs;
                current->ctx.rip = regs->rip;
                current->ctx.rflags = regs->rflags;
            }

            triad_sched_do_switch();

            TriadThread *next = triad_sched_current();
            if (next && next->ctx.rsp != 0) {
                regs = (TriadRegisters *)next->ctx.rsp;
            }
        }

        triad_isr_eoi(IRQ_TIMER);
    } else if (int_no == IRQ_KEYBOARD) {
        if (callbacks[int_no]) {
            callbacks[int_no](regs);
        }
        triad_isr_eoi(IRQ_KEYBOARD);
    } else if (int_no == IRQ_SERIAL) {
        if (callbacks[int_no]) {
            callbacks[int_no](regs);
        }
        triad_isr_eoi(IRQ_SERIAL);
    } else if (int_no == IRQ_SYSCALL) {
        triad_kpti_switch_to_kernel();
        int64_t result = triad_syscall_dispatch(
            (TriadSyscall)regs->rax,
            regs->rbx, regs->rcx, regs->rdx,
            regs->rsi, regs->rdi
        );
        regs->rax = (uint64_t)result;
        triad_kpti_switch_to_user();
    } else if (int_no >= 32 && int_no < 48) {
        if (callbacks[int_no]) {
            callbacks[int_no](regs);
        }
        triad_isr_eoi((int)int_no);
    }
    return regs;
}

void isr_table_init(void) {
    extern uint64_t isr_0, isr_1, isr_2, isr_3, isr_4, isr_5, isr_6, isr_7;
    extern uint64_t isr_8, isr_9, isr_10, isr_11, isr_12, isr_13, isr_14, isr_15;
    extern uint64_t isr_16, isr_17, isr_18, isr_19, isr_20, isr_21, isr_22, isr_23;
    extern uint64_t isr_24, isr_25, isr_26, isr_27, isr_28, isr_29, isr_30, isr_31;
    extern uint64_t isr_32, isr_33, isr_34, isr_35, isr_36, isr_37, isr_38, isr_39;
    extern uint64_t isr_40, isr_41, isr_42, isr_43, isr_44, isr_45, isr_46, isr_47;
    extern uint64_t isr_128;

    isr_table[0]  = (uint64_t)&isr_0;
    isr_table[1]  = (uint64_t)&isr_1;
    isr_table[2]  = (uint64_t)&isr_2;
    isr_table[3]  = (uint64_t)&isr_3;
    isr_table[4]  = (uint64_t)&isr_4;
    isr_table[5]  = (uint64_t)&isr_5;
    isr_table[6]  = (uint64_t)&isr_6;
    isr_table[7]  = (uint64_t)&isr_7;
    isr_table[8]  = (uint64_t)&isr_8;
    isr_table[9]  = (uint64_t)&isr_9;
    isr_table[10] = (uint64_t)&isr_10;
    isr_table[11] = (uint64_t)&isr_11;
    isr_table[12] = (uint64_t)&isr_12;
    isr_table[13] = (uint64_t)&isr_13;
    isr_table[14] = (uint64_t)&isr_14;
    isr_table[15] = (uint64_t)&isr_15;
    isr_table[16] = (uint64_t)&isr_16;
    isr_table[17] = (uint64_t)&isr_17;
    isr_table[18] = (uint64_t)&isr_18;
    isr_table[19] = (uint64_t)&isr_19;
    isr_table[20] = (uint64_t)&isr_20;
    isr_table[21] = (uint64_t)&isr_21;
    isr_table[22] = (uint64_t)&isr_22;
    isr_table[23] = (uint64_t)&isr_23;
    isr_table[24] = (uint64_t)&isr_24;
    isr_table[25] = (uint64_t)&isr_25;
    isr_table[26] = (uint64_t)&isr_26;
    isr_table[27] = (uint64_t)&isr_27;
    isr_table[28] = (uint64_t)&isr_28;
    isr_table[29] = (uint64_t)&isr_29;
    isr_table[30] = (uint64_t)&isr_30;
    isr_table[31] = (uint64_t)&isr_31;
    isr_table[32] = (uint64_t)&isr_32;
    isr_table[33] = (uint64_t)&isr_33;
    isr_table[34] = (uint64_t)&isr_34;
    isr_table[35] = (uint64_t)&isr_35;
    isr_table[36] = (uint64_t)&isr_36;
    isr_table[37] = (uint64_t)&isr_37;
    isr_table[38] = (uint64_t)&isr_38;
    isr_table[39] = (uint64_t)&isr_39;
    isr_table[40] = (uint64_t)&isr_40;
    isr_table[41] = (uint64_t)&isr_41;
    isr_table[42] = (uint64_t)&isr_42;
    isr_table[43] = (uint64_t)&isr_43;
    isr_table[44] = (uint64_t)&isr_44;
    isr_table[45] = (uint64_t)&isr_45;
    isr_table[46] = (uint64_t)&isr_46;
    isr_table[47] = (uint64_t)&isr_47;
    isr_table[128] = (uint64_t)&isr_128;
}