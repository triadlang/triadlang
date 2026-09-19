#include "triad_user.h"
#include "triad_kernel.h"
#include "triad_mm.h"
#include "triad_paging.h"
#include "triad_gdt_idt.h"
#include "triad_serial.h"
#include <string.h>

static UserProc user_procs[16];
static int user_proc_count = 0;
static int current_user_proc = -1;
static uint8_t *tss_mem = NULL;

static void tss_init(void) {
    tss_mem = (uint8_t *)triad_mm_alloc_aligned(0x68, 16);
    memset(tss_mem, 0, 0x68);
    *(uint16_t *)(tss_mem + 0x66) = 0x68;
}

static void gdt_set_tss(int sel, uint64_t base, uint64_t limit) {
    uint64_t *entry = (uint64_t *)&gdt[sel / 8];
    uint64_t lo = 0;
    lo |= (uint64_t)(limit & 0xFFFF);
    lo |= (uint64_t)(base & 0xFFFFFFULL) << 16;
    lo |= (uint64_t)0x89 << 40;
    lo |= (uint64_t)((limit >> 16) & 0xFULL) << 48;
    lo |= (uint64_t)((base >> 24) & 0xFFULL) << 56;
    entry[0] = lo;
    entry[1] = (base >> 32) & 0xFFFFFFFFULL;
}

static void load_tss(void) {
    uint64_t tss_base = (uint64_t)tss_mem;
    gdt_set_tss(0x28, tss_base, 0x68);
    __asm__ volatile ("ltr %w0" : : "r"((uint16_t)0x28));
}

static void user_segments_init(void) {
    uint64_t *user_cs = (uint64_t *)&gdt[3];
    uint64_t *user_ds = (uint64_t *)&gdt[4];
    *user_cs = 0x00AFFFFF0000FFFFULL;
    *user_ds = 0x00CFFFFF0000FFFFULL;
}

void triad_user_init(void) {
    for (int i = 0; i < 16; i++) {
        user_procs[i].active = false;
        user_procs[i].id = -1;
    }
    user_proc_count = 0;
    current_user_proc = -1;
    tss_init();
    user_segments_init();
    load_tss();
}

int triad_user_create(const char *name, uint64_t entry) {
    int idx = -1;
    for (int i = 0; i < 16; i++) {
        if (!user_procs[i].active) {
            idx = i;
            break;
        }
    }
    if (idx < 0) return -1;

    UserProc *proc = &user_procs[idx];
    proc->id = idx;
    proc->entry = entry;
    proc->active = true;

    int k = 0;
    while (name[k] && k < 63) {
        proc->name[k] = name[k];
        k++;
    }
    proc->name[k] = 0;

    uint64_t user_stack = USER_ENTRY_POINT + USER_STACK_SIZE;
    proc->stack_top = user_stack;
    proc->brk = USER_ENTRY_POINT + 0x100000;

    for (uint64_t addr = USER_ENTRY_POINT; addr < proc->brk; addr += 0x1000) {
        void *page = triad_paging_alloc_page();
        if (page) {
            triad_paging_map(addr, (uint64_t)page, PTE_PRESENT | PTE_WRITABLE | PTE_USER);
        }
    }

    for (uint64_t addr = USER_ENTRY_POINT + USER_STACK_SIZE - 0x10000;
         addr < USER_ENTRY_POINT + USER_STACK_SIZE;
         addr += 0x1000) {
        void *page = triad_paging_alloc_page();
        if (page) {
            triad_paging_map(addr, (uint64_t)page, PTE_PRESENT | PTE_WRITABLE | PTE_USER);
        }
    }

    user_proc_count++;
    return idx;
}

void triad_user_switch(int proc_id) {
    if (proc_id < 0 || proc_id >= 16) return;
    if (!user_procs[proc_id].active) return;

    UserProc *proc = &user_procs[proc_id];
    current_user_proc = proc_id;

    uint64_t rsp0 = proc->stack_top;
    uint8_t *tss_ptr = tss_mem;
    tss_ptr[0x04] = (uint8_t)(rsp0 & 0xFF);
    tss_ptr[0x05] = (uint8_t)((rsp0 >> 8) & 0xFF);
    tss_ptr[0x06] = (uint8_t)((rsp0 >> 16) & 0xFF);
    tss_ptr[0x07] = (uint8_t)((rsp0 >> 24) & 0xFF);
    tss_ptr[0x08] = (uint8_t)((rsp0 >> 32) & 0xFF);
    tss_ptr[0x09] = (uint8_t)((rsp0 >> 40) & 0xFF);
    tss_ptr[0x0A] = (uint8_t)((rsp0 >> 48) & 0xFF);
    tss_ptr[0x0B] = (uint8_t)((rsp0 >> 56) & 0xFF);

    uint64_t ds = USER_DS | 3;
    uint64_t ss = USER_DS | 3;

    __asm__ volatile (
        "mov %0, %%ds\n"
        "mov %0, %%es\n"
        "mov %0, %%fs\n"
        "mov %0, %%gs\n"
        :
        : "a"(ds)
    );

    uint64_t rip = proc->entry;
    uint64_t rsp = proc->stack_top;
    uint64_t rflags = 0x202;

    __asm__ volatile (
        "cli\n"
        "mov %2, %%rsp\n"
        "push %1\n"
        "push %0\n"
        "push %3\n"
        "push %2\n"
        "iretq\n"
        :
        : "r"(rip), "r"(ss), "r"(rsp), "r"(rflags)
        : "memory"
    );
}

void triad_user_exit(int proc_id) {
    if (proc_id < 0 || proc_id >= 16) return;
    user_procs[proc_id].active = false;
    if (user_proc_count > 0) user_proc_count--;
    if (current_user_proc == proc_id) {
        current_user_proc = -1;
    }
}

bool triad_user_is_user_mode(void) {
    uint64_t cs;
    __asm__ volatile ("mov %%cs, %0" : "=r"(cs));
    return (cs & 3) == 3;
}