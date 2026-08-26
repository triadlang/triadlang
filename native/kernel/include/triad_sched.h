#ifndef TRIAD_SCHED_H
#define TRIAD_SCHED_H

#include "triad_kernel.h"

void triad_sched_init(void);
void triad_sched_tick(void);
void triad_sched_yield(void);
void triad_sched_block(int thread_id);
void triad_sched_unblock(int thread_id);
int  triad_sched_spawn(TriadThread *t);
void triad_sched_exit(int thread_id);
TriadThread *triad_sched_current(void);

int  triad_thread_create(TriadThread *t, void (*entry)(void *), void *arg);
void triad_sched_do_switch(void);
int  triad_sched_needs_switch(void);
void triad_sched_save_ctx(TriadSavedCtx *ctx);
void triad_sched_restore_ctx(TriadSavedCtx *ctx);
uint64_t triad_sched_get_ticks(void);
int triad_sched_thread_count(void);

#endif