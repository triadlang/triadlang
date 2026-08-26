#ifndef TRIAD_RUNTIME_SVC_H
#define TRIAD_RUNTIME_SVC_H

#include "triad_kernel.h"

#define RT_SVC_MAX_PROGRAMS 16

typedef struct {
    bool     active;
    int      proc_id;
    int      thread_id;
    char     source_path[256];
    uint64_t start_tick;
    uint64_t end_tick;
    int      exit_code;
} RtProgram;

void triad_rt_svc_init(void);
int  triad_rt_svc_load(const char *vfs_path, int proc_id);
int  triad_rt_svc_run(int program_id);
int  triad_rt_svc_status(int program_id, RtProgram *out);
int  triad_rt_svc_list(RtProgram *out, int max);
void triad_rt_svc_observe_all(void);

#endif