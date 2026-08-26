#ifndef TRIAD_USER_H
#define TRIAD_USER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define USER_CS     0x18
#define USER_DS     0x20
#define USER_SS     0x20
#define USER_STACK_SIZE  (64 * 1024)
#define USER_ENTRY_POINT 0x400000

typedef struct UserCtx {
    uint64_t r15, r14, r13, r12, r11, r10, r9, r8;
    uint64_t rdi, rsi, rbp, rdx, rcx, rbx, rax;
    uint64_t rip, rsp, rflags;
    uint64_t cs, ds, ss;
} UserCtx;

typedef struct UserProc {
    int id;
    char name[64];
    uint64_t entry;
    uint64_t stack_top;
    uint64_t brk;
    bool active;
    uint8_t *elf_image;
    size_t elf_size;
} UserProc;

void triad_user_init(void);
int triad_user_create(const char *name, uint64_t entry);
void triad_user_switch(int proc_id);
void triad_user_exit(int proc_id);
bool triad_user_is_user_mode(void);

#endif