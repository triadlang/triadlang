#ifndef TRIAD_CONTEXT_SWITCH_H
#define TRIAD_CONTEXT_SWITCH_H

#include <stdint.h>
#include "triad_kernel.h"

void triad_ctx_save(TriadSavedCtx *ctx);
void triad_ctx_restore(TriadSavedCtx *ctx);
void triad_ctx_switch(TriadSavedCtx *from, TriadSavedCtx *to);
void triad_ctx_init_thread(TriadSavedCtx *ctx, uint64_t entry, uint64_t arg);

#endif