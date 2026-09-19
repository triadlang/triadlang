#include "qos_integration.h"
#include "triad_kernel.h"
#include "triad_vfs.h"

static int64_t syscall_dispatch_extended(uint64_t num, uint64_t a, uint64_t b,
                                          uint64_t c, uint64_t d, uint64_t e) {
    switch (num) {
        case SYS_EXIT:
            return 0;
        case SYS_WRITE:
            return triad_vfs_write((int)a, (const void *)b, (uint32_t)c);
        case SYS_READ:
            return triad_vfs_read((int)a, (void *)b, (uint32_t)c);
        case SYS_OPEN:
            return triad_vfs_open((const char *)a, (uint8_t)b);
        case SYS_CLOSE:
            return triad_vfs_close((int)a);
        case SYS_SEEK:
            return triad_vfs_seek((int)a, (uint32_t)b);
        case SYS_STAT:
            return triad_vfs_stat((const char *)a, (TriadVfsNode *)b);
        case SYS_MKDIR:
            return triad_vfs_mkdir((const char *)a);
        case SYS_LISTDIR:
            return triad_vfs_listdir((const char *)a, (char (*)[256])b, (int)c);
        case SYS_REMOVE:
            return triad_vfs_remove((const char *)a);
        case SYS_SPAWN:
            return qos_proc_from_tri_file((const char *)a);
        case SYS_WAIT:
            if (a >= 0 && a < MAX_PROCS) {
                qos_wait_equilibrium((int)a, 0.01);
                return 0;
            }
            return -1;
        case SYS_GETPID:
            return qos_sched_get_current();
        case SYS_GETTICKS:
            return (int64_t)(g_qos_sched.global_time * 1000);
        case SYS_SLEEP:
            qos_evolve_all((double)a / 1000.0);
            return 0;
        case SYS_YIELD:
            qos_sched_yield();
            return 0;
        case SYS_QUANTUM_CREATE:
            return sys_quantum_create(a, b, c);
        case SYS_QUANTUM_EVOLVE:
            return sys_quantum_evolve(a, b);
        case SYS_QUANTUM_MEASURE:
            return sys_quantum_measure(a, b);
        case SYS_QUANTUM_COUPLE:
            return sys_quantum_couple(a, b, c);
        case SYS_QUANTUM_DECOUPLE:
            qos_decouple((int)a, (int)b);
            return 0;
        case SYS_QUANTUM_SAVE:
            return qos_vfs_save_quantum_state((const char *)a, (int)b);
        case SYS_QUANTUM_LOAD:
            return qos_vfs_load_quantum_state((const char *)a, (int)b);
        case SYS_LLM_GENERATE:
            return sys_llm_generate(a, b, c);
        case SYS_LLM_ABSORB:
            return sys_llm_absorb(a, b);
        case SYS_LLM_SUGGEST:
            return qos_llm_suggest((const char *)a, (char *)b);
        case SYS_AI_QUERY:
            return -1;
        case SYS_AI_RECORD:
            qos_llm_absorb_text((const char *)a);
            return 0;
        case SYS_SOLVER:
            qos_evolve_all((double)a);
            return 0;
        case SYS_EQUILIBRIUM:
            return g_qos_sched.quantum_procs[a].crystallized;
        default:
            return -1;
    }
}

enum {
    SYS_QUANTUM_CREATE = 50,
    SYS_QUANTUM_EVOLVE = 51,
    SYS_QUANTUM_MEASURE = 52,
    SYS_QUANTUM_COUPLE = 53,
    SYS_QUANTUM_DECOUPLE = 54,
    SYS_QUANTUM_SAVE = 55,
    SYS_QUANTUM_LOAD = 56,
    SYS_LLM_GENERATE = 60,
    SYS_LLM_ABSORB = 61,
    SYS_LLM_SUGGEST = 62,
};
