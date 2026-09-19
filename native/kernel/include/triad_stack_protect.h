#ifndef TRIAD_STACK_PROTECT_H
#define TRIAD_STACK_PROTECT_H

#include <stdint.h>
#include <stddef.h>

extern uint64_t __stack_chk_guard;

uint64_t __stack_chk_setup(void);
void __stack_chk_fail(void) __attribute__((noreturn));

void triad_stack_canary_init(void);
uint64_t triad_stack_random_canary(void);

typedef struct {
    uint64_t canary;
    uint64_t ret_addr;
    uint64_t saved_rbp;
} StackFrame;

int triad_stack_check_frame(StackFrame *frame);
void triad_stack_panic(const char *msg) __attribute__((noreturn));

#endif