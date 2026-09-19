#include "triad_runtime_svc.h"
#include "triad_vfs.h"
#include "triad_mm.h"
#include "triad_ai.h"
#include "triad_sched.h"
#include "triad_serial.h"
#include "triad_syscall.h"
#include "triad_interpreter.h"
#include <string.h>

static RtProgram programs[RT_SVC_MAX_PROGRAMS];

void triad_rt_svc_init(void) {
    for (int i = 0; i < RT_SVC_MAX_PROGRAMS; i++) {
        programs[i].active = false;
    }
}

int triad_rt_svc_load(const char *vfs_path, int proc_id) {
    int idx = -1;
    for (int i = 0; i < RT_SVC_MAX_PROGRAMS; i++) {
        if (!programs[i].active) {
            idx = i;
            break;
        }
    }
    if (idx < 0) return -1;

    TriadVfsNode node;
    if (triad_vfs_stat(vfs_path, &node) < 0) {
        triad_serial_puts("[rt_svc] file not found: ");
        triad_serial_puts(vfs_path);
        triad_serial_puts("\n");
        return -1;
    }

    programs[idx].active = true;
    programs[idx].proc_id = proc_id;
    int k = 0;
    while (vfs_path[k] && k < 255) {
        programs[idx].source_path[k] = vfs_path[k];
        k++;
    }
    programs[idx].source_path[k] = 0;
    programs[idx].start_tick = 0;
    programs[idx].end_tick = 0;
    programs[idx].exit_code = 0;

    char observe_msg[300];
    const char prefix[] = "loaded program: ";
    int j = 0;
    while (prefix[j] && j < 299) { observe_msg[j] = prefix[j]; j++; }
    k = 0;
    while (vfs_path[k] && j < 299) { observe_msg[j] = vfs_path[k]; j++; k++; }
    observe_msg[j] = 0;
    triad_ai_observe(proc_id, "rt_svc", observe_msg);

    return idx;
}

int triad_rt_svc_run(int program_id) {
    if (program_id < 0 || program_id >= RT_SVC_MAX_PROGRAMS) return -1;
    if (!programs[program_id].active) return -1;

    programs[program_id].start_tick = triad_sched_get_ticks();

    int fd = triad_vfs_open(programs[program_id].source_path, 0);
    if (fd < 0) {
        programs[program_id].exit_code = -1;
        return -1;
    }

    char buf[4096];
    int n = triad_vfs_read(fd, buf, sizeof(buf) - 1);
    triad_vfs_close(fd);
    if (n > 0) {
        buf[n] = 0;
        triad_ai_observe(programs[program_id].proc_id,
                         programs[program_id].source_path, buf);
        TriadInterp *interp = (TriadInterp *)triad_mm_alloc(sizeof(TriadInterp));
        if (!interp) {
            programs[program_id].exit_code = -1;
            return -1;
        }
        triad_interp_init(interp, programs[program_id].proc_id);
        triad_interp_run(interp, buf);
    }

    programs[program_id].end_tick = triad_sched_get_ticks();
    programs[program_id].exit_code = 0;

    return 0;
}

int triad_rt_svc_status(int program_id, RtProgram *out) {
    if (program_id < 0 || program_id >= RT_SVC_MAX_PROGRAMS) return -1;
    if (!programs[program_id].active) return -1;
    *out = programs[program_id];
    return 0;
}

int triad_rt_svc_list(RtProgram *out, int max) {
    int count = 0;
    for (int i = 0; i < RT_SVC_MAX_PROGRAMS && count < max; i++) {
        if (programs[i].active) {
            out[count] = programs[i];
            count++;
        }
    }
    return count;
}

void triad_rt_svc_observe_all(void) {
    for (int i = 0; i < RT_SVC_MAX_PROGRAMS; i++) {
        if (programs[i].active) {
            char msg[300];
            const char prefix[] = "program active: ";
            int j = 0;
            while (prefix[j] && j < 299) { msg[j] = prefix[j]; j++; }
            int k = 0;
            while (programs[i].source_path[k] && j < 299) {
                msg[j] = programs[i].source_path[k]; j++; k++;
            }
            msg[j] = 0;
            triad_ai_observe(programs[i].proc_id, "rt_svc", msg);
        }
    }
}