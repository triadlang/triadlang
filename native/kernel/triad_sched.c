#include "triad_kernel.h"
#include "triad_sched.h"
#include "triad_mm.h"
#include "triad_serial.h"

static TriadThread threads[MAX_THREADS];
static int current_tid = 0;
static int n_threads = 0;
static uint64_t global_ticks = 0;
static volatile int needs_context_switch = 0;

void triad_sched_init(void) {
    for (int i = 0; i < MAX_THREADS; i++) {
        threads[i].state = THREAD_DEAD;
        threads[i].stack_base = NULL;
        threads[i].stack_size = 0;
    }
    current_tid = 0;
    n_threads = 0;
    global_ticks = 0;
    needs_context_switch = 0;

    threads[0].id = 0;
    threads[0].proc_id = 0;
    threads[0].state = THREAD_RUNNING;
    threads[0].stack_base = NULL;
    threads[0].stack_size = 0;
    threads[0].quantum_ticks = TIMER_HZ / 10;
    threads[0].total_ticks = 0;
    threads[0].triad_closure = NULL;
    n_threads = 1;
}

TriadThread *triad_sched_current(void) {
    if (current_tid < 0 || current_tid >= MAX_THREADS) return NULL;
    return &threads[current_tid];
}

int triad_sched_spawn(TriadThread *t) {
    if (!t) return -1;
    for (int i = 0; i < MAX_THREADS; i++) {
        if (threads[i].state == THREAD_DEAD) {
            threads[i] = *t;
            threads[i].id = i;
            threads[i].state = THREAD_READY;
            threads[i].quantum_ticks = TIMER_HZ / 10;
            threads[i].total_ticks = 0;
            n_threads++;
            return i;
        }
    }
    return -1;
}

static int find_next_thread(void) {
    int start = current_tid;
    for (int i = 1; i <= MAX_THREADS; i++) {
        int next = (start + i) % MAX_THREADS;
        if (threads[next].state == THREAD_READY) {
            return next;
        }
    }
    return -1;
}

void triad_sched_tick(void) {
    global_ticks++;

    TriadThread *current = &threads[current_tid];
    if (current->state != THREAD_RUNNING) {
        needs_context_switch = 1;
        return;
    }

    current->quantum_ticks--;
    if (current->quantum_ticks == 0) {
        current->quantum_ticks = TIMER_HZ / 10;
        int next = find_next_thread();
        if (next >= 0 && next != current_tid) {
            current->state = THREAD_READY;
            needs_context_switch = 1;
        }
    }
}

int triad_sched_needs_switch(void) {
    return needs_context_switch;
}

void triad_sched_yield(void) {
    int next = find_next_thread();
    if (next >= 0 && next != current_tid) {
        if (threads[current_tid].state == THREAD_RUNNING)
            threads[current_tid].state = THREAD_READY;
        needs_context_switch = 1;
    }
}

void triad_sched_block(int thread_id) {
    if (thread_id < 0 || thread_id >= MAX_THREADS) return;
    threads[thread_id].state = THREAD_BLOCKED;
    if (thread_id == current_tid) {
        needs_context_switch = 1;
    }
}

void triad_sched_unblock(int thread_id) {
    if (thread_id < 0 || thread_id >= MAX_THREADS) return;
    if (threads[thread_id].state == THREAD_BLOCKED) {
        threads[thread_id].state = THREAD_READY;
    }
}

void triad_sched_exit(int thread_id) {
    if (thread_id < 0 || thread_id >= MAX_THREADS) return;
    threads[thread_id].state = THREAD_DEAD;
    if (n_threads > 0) n_threads--;
    if (thread_id == current_tid) {
        needs_context_switch = 1;
    }
}

static void thread_entry_wrapper(void (*entry)(void *), void *arg) {
    entry(arg);
    triad_sched_exit(current_tid);
    while (1) { __asm__ volatile ("hlt"); }
}

int triad_thread_create(TriadThread *t, void (*entry)(void *), void *arg) {
    uint64_t *stack = (uint64_t *)triad_mm_alloc(STACK_SIZE);
    if (!stack) return -1;

    t->stack_base = stack;
    t->stack_size = STACK_SIZE;
    t->state = THREAD_READY;
    t->quantum_ticks = TIMER_HZ / 10;
    t->total_ticks = 0;

    uint64_t *sp = (uint64_t *)((uint8_t *)stack + STACK_SIZE);
    *--sp = 0x10;
    *--sp = (uint64_t)((uint8_t *)stack + STACK_SIZE) - 8;
    *--sp = 0x202;
    *--sp = 0x08;
    *--sp = (uint64_t)thread_entry_wrapper;
    *--sp = 0;
    *--sp = 0;
    *--sp = 0;
    *--sp = 0;
    *--sp = 0;
    *--sp = 0;
    *--sp = 0;
    *--sp = (uint64_t)arg;
    *--sp = (uint64_t)entry;
    *--sp = 0;
    *--sp = 0;
    *--sp = 0;
    *--sp = 0;
    *--sp = 0;
    *--sp = 0;
    *--sp = 0;
    *--sp = 0;

    t->ctx.rsp = (uint64_t)sp;
    t->ctx.rip = (uint64_t)thread_entry_wrapper;
    t->ctx.rflags = 0x202;

    return triad_sched_spawn(t);
}

void triad_sched_save_ctx(TriadSavedCtx *ctx) {
    __asm__ volatile (
        "mov %%r15, %0\n"
        "mov %%r14, %1\n"
        "mov %%r13, %2\n"
        "mov %%r12, %3\n"
        "mov %%r11, %4\n"
        "mov %%r10, %5\n"
        "mov %%r9, %6\n"
        "mov %%r8, %7\n"
        : "=m"(ctx->r15), "=m"(ctx->r14), "=m"(ctx->r13), "=m"(ctx->r12),
          "=m"(ctx->r11), "=m"(ctx->r10), "=m"(ctx->r9), "=m"(ctx->r8)
    );
    __asm__ volatile (
        "mov %%rdi, %0\n"
        "mov %%rsi, %1\n"
        "mov %%rbp, %2\n"
        "mov %%rdx, %3\n"
        "mov %%rcx, %4\n"
        "mov %%rbx, %5\n"
        "mov %%rax, %6\n"
        : "=m"(ctx->rdi), "=m"(ctx->rsi), "=m"(ctx->rbp), "=m"(ctx->rdx),
          "=m"(ctx->rcx), "=m"(ctx->rbx), "=m"(ctx->rax)
    );
}

void triad_sched_restore_ctx(TriadSavedCtx *ctx) {
    __asm__ volatile (
        "mov %0, %%r15\n"
        "mov %1, %%r14\n"
        "mov %2, %%r13\n"
        "mov %3, %%r12\n"
        : : "m"(ctx->r15), "m"(ctx->r14), "m"(ctx->r13), "m"(ctx->r12)
    );
    __asm__ volatile (
        "mov %0, %%r11\n"
        "mov %1, %%r10\n"
        "mov %2, %%r9\n"
        "mov %3, %%r8\n"
        : : "m"(ctx->r11), "m"(ctx->r10), "m"(ctx->r9), "m"(ctx->r8)
    );
    __asm__ volatile (
        "mov %0, %%rdi\n"
        "mov %1, %%rsi\n"
        "mov %2, %%rbp\n"
        "mov %3, %%rdx\n"
        "mov %4, %%rcx\n"
        "mov %5, %%rbx\n"
        "mov %6, %%rax\n"
        : : "m"(ctx->rdi), "m"(ctx->rsi), "m"(ctx->rbp), "m"(ctx->rdx),
            "m"(ctx->rcx), "m"(ctx->rbx), "m"(ctx->rax)
    );
}

void triad_sched_do_switch(void) {
    if (!needs_context_switch) return;

    int next = find_next_thread();
    if (next < 0 || next == current_tid) {
        if (threads[current_tid].state != THREAD_DEAD &&
            threads[current_tid].state != THREAD_BLOCKED)
            threads[current_tid].state = THREAD_RUNNING;
        needs_context_switch = 0;
        return;
    }

    if (threads[current_tid].state == THREAD_RUNNING)
        threads[current_tid].state = THREAD_READY;
    current_tid = next;
    threads[next].state = THREAD_RUNNING;

    needs_context_switch = 0;
}

uint64_t triad_sched_get_ticks(void) {
    return global_ticks;
}

int triad_sched_thread_count(void) {
    return n_threads;
}