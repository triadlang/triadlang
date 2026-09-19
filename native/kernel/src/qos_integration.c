#include "qos_integration.h"
#include "qos_process.h"
#include "qos_llm.h"
#include "triad_kernel.h"
#include "triad_sched.h"
#include "triad_vfs.h"
#include <stddef.h>
#include <string.h>

static QosScheduler g_qos_sched;
static int g_qos_initialized = 0;

void qos_kernel_init(void) {
    if (g_qos_initialized) return;
    memset(&g_qos_sched, 0, sizeof(QosScheduler));
    g_qos_sched.n_classic = 0;
    g_qos_sched.n_quantum = 0;
    g_qos_sched.global_time = 0;
    g_qos_sched.dt = 0.001;
    g_qos_sched.quantum_enabled = 1;
    g_qos_sched.llm_enabled = 1;
    qos_init();
    g_qos_sched.qos_rt = NULL;
    qos_llm_init();
    g_qos_sched.kernel_llm = NULL;
    g_qos_initialized = 1;
}

void qos_scheduler_init(void) {
    qos_kernel_init();
}

int qos_proc_create(const char *name, int is_quantum) {
    if (!g_qos_initialized) qos_kernel_init();
    int pid = -1;
    if (is_quantum) {
        if (g_qos_sched.n_quantum >= MAX_PROCS) return -1;
        QosProc *qp = &g_qos_sched.quantum_procs[g_qos_sched.n_quantum];
        memset(qp, 0, sizeof(QosProc));
        qp->base.id = g_qos_sched.n_quantum;
        strncpy(qp->base.name, name, sizeof(qp->base.name) - 1);
        qp->base.alive = 1;
        qp->base.tier = PROC_T2_SIGNED;
        qp->quantum = NULL;
        qp->llm_ctx = NULL;
        qp->crystallized = 0;
        qp->crystallinity = 0.0;
        pid = g_qos_sched.n_quantum;
        g_qos_sched.n_quantum++;
    } else {
        if (g_qos_sched.n_classic >= MAX_PROCS) return -1;
        TriadProc *tp = &g_qos_sched.classic_procs[g_qos_sched.n_classic];
        memset(tp, 0, sizeof(TriadProc));
        tp->id = g_qos_sched.n_classic;
        strncpy(tp->name, name, sizeof(tp->name) - 1);
        tp->alive = 1;
        tp->tier = PROC_T3_USER;
        pid = g_qos_sched.n_classic;
        g_qos_sched.n_classic++;
    }
    return pid;
}

int qos_proc_from_tri_file(const char *path) {
    if (!g_qos_initialized) qos_kernel_init();
    int is_quantum = 0;
    size_t len = strlen(path);
    if (len > 4 && strcmp(path + len - 4, ".tri") == 0) {
        is_quantum = 1;
    }
    int pid = qos_proc_create(path, is_quantum);
    if (pid < 0) return -1;
    if (is_quantum) {
        QosProc *qp = &g_qos_sched.quantum_procs[pid];
        strncpy(qp->source_file, path, sizeof(qp->source_file) - 1);
        int substrate_id = qos_create_process(path, 256, 1, 10.0);
        if (substrate_id >= 0) {
        }
    }
    return pid;
}

void qos_sched_tick(void) {
    if (!g_qos_initialized) return;
    qos_evolve_all(g_qos_sched.dt);
    for (int i = 0; i < g_qos_sched.n_quantum; i++) {
        QosProc *qp = &g_qos_sched.quantum_procs[i];
        if (qp->quantum) {
            qp->crystallinity = qos_measure_crystallinity(qp->quantum->pid);
            if (qp->crystallinity > 0.95) {
                qp->crystallized = 2;
            } else if (qp->crystallinity > 0.8) {
                qp->crystallized = 1;
            } else {
                qp->crystallized = 0;
            }
        }
    }
    g_qos_sched.global_time += g_qos_sched.dt;
    if (g_qos_sched.llm_enabled) {
        for (int i = 0; i < g_qos_sched.n_quantum; i++) {
            QosProc *qp = &g_qos_sched.quantum_procs[i];
            if (qp->quantum && qp->crystallized >= 1) {
                qos_llm_absorb_process(g_qos_sched.kernel_llm, qp->quantum);
            }
        }
    }
}

void qos_sched_yield(void) {
    qos_sched_tick();
}

int qos_sched_get_current(void) {
    for (int i = 0; i < g_qos_sched.n_quantum; i++) {
        if (g_qos_sched.quantum_procs[i].base.alive) {
            return i;
        }
    }
    return -1;
}

int64_t sys_quantum_create(uint64_t N, uint64_t L, uint64_t Lambda) {
    if (!g_qos_initialized) qos_kernel_init();
    char name[64];
    snprintf(name, sizeof(name), "quantum_%d", g_qos_sched.n_quantum);
    int substrate_id = qos_create_process(name, (int)N, 1, (double)L);
    if (substrate_id < 0) return -1;
    qos_set_parameters(substrate_id, (double)Lambda, 0.1, 0.01);
    return substrate_id;
}

int64_t sys_quantum_evolve(uint64_t pid, uint64_t T) {
    if (!g_qos_initialized) return -1;
    qos_evolve_all((double)T);
    return 0;
}

int64_t sys_quantum_measure(uint64_t pid, uint64_t observable) {
    if (!g_qos_initialized) return -1;
    int id = (int)pid;
    switch (observable) {
        case 0:
            return (int64_t)(qos_measure_energy(id) * 1000);
        case 1:
            return (int64_t)(qos_measure_crystallinity(id) * 1000);
        case 2:
            if (id >= 0 && id < g_qos_sched.n_quantum) {
                return g_qos_sched.quantum_procs[id].crystallized;
            }
            return 0;
        default:
            return 0;
    }
}

int64_t sys_quantum_couple(uint64_t src, uint64_t dst, uint64_t kappa) {
    if (!g_qos_initialized) return -1;
    qos_couple((int)src, (int)dst, (double)kappa / 1000.0);
    return 0;
}

int64_t sys_llm_generate(uint64_t prompt_ptr, uint64_t out_ptr, uint64_t max_len) {
    if (!g_qos_initialized) return -1;
    (void)prompt_ptr;
    (void)out_ptr;
    (void)max_len;
    return -1;
}

int64_t sys_llm_absorb(uint64_t data_ptr, uint64_t len) {
    if (!g_qos_initialized) return -1;
    (void)data_ptr;
    (void)len;
    return -1;
}

int qos_vfs_save_quantum_state(const char *path, int pid) {
    if (!g_qos_initialized) return -1;
    if (pid < 0 || pid >= g_qos_sched.n_quantum) return -1;
    QosProc *qp = &g_qos_sched.quantum_procs[pid];
    if (!qp->quantum) return -1;
    int fd = triad_vfs_create(path);
    if (fd < 0) return -1;
    triad_vfs_close(fd);
    return 0;
}

int qos_vfs_load_quantum_state(const char *path, int pid) {
    if (!g_qos_initialized) return -1;
    (void)path;
    (void)pid;
    return -1;
}

int qos_eval_tri_code(const char *code, char *output, int max_len) {
    if (!g_qos_initialized) qos_kernel_init();
    qos_llm_absorb_text(code);
    int pid = qos_proc_create("eval", 1);
    if (pid < 0) {
        if (output && max_len > 0) {
            strncpy(output, "Error: Failed to create process", max_len - 1);
        }
        return -1;
    }
    qos_evolve_all(1.0);
    QosProc *qp = &g_qos_sched.quantum_procs[pid];
    if (qp->crystallized >= 1) {
        if (output && max_len > 0) {
            strncpy(output, "[OK] Code crystallized", max_len - 1);
        }
        return 0;
    } else {
        if (output && max_len > 0) {
            strncpy(output, "[...] Still evolving", max_len - 1);
        }
        return 1;
    }
}

int qos_run_tri_file(const char *path) {
    if (!g_qos_initialized) qos_kernel_init();
    int pid = qos_proc_from_tri_file(path);
    if (pid < 0) return -1;
    qos_evolve_all(1.0);
    return pid;
}

void qos_shell_run(void) {
    if (!g_qos_initialized) qos_kernel_init();
    extern void qos_shell_start(void);
    qos_shell_start();
}

int qos_shell_execute(const char *cmd) {
    if (!g_qos_initialized) qos_kernel_init();
    qos_llm_absorb_text(cmd);
    if (strncmp(cmd, "ls", 2) == 0) {
        return 0;
    } else if (strncmp(cmd, "run ", 4) == 0) {
        const char *path = cmd + 4;
        return qos_run_tri_file(path);
    } else if (strncmp(cmd, "eval ", 5) == 0) {
        const char *code = cmd + 5;
        char output[256];
        return qos_eval_tri_code(code, output, sizeof(output));
    } else if (strncmp(cmd, "status", 6) == 0) {
        return 0;
    } else if (strncmp(cmd, "mem", 3) == 0) {
        return 0;
    } else if (strncmp(cmd, "ps", 2) == 0) {
        return 0;
    }
    return -1;
}

void qos_llm_absorb_process(int pid) {
    if (!g_qos_initialized) return;
    if (pid < 0 || pid >= g_qos_sched.n_quantum) return;
    QosProc *qp = &g_qos_sched.quantum_procs[pid];
    if (qp->quantum) {
        qos_llm_absorb_process(g_qos_sched.kernel_llm, qp->quantum);
    }
}

void qos_llm_absorb_text(const char *text) {
    if (!g_qos_initialized) return;
    qos_llm_absorb(g_qos_sched.kernel_llm, text, strlen(text));
}

int qos_llm_suggest(const char *context, char *suggestion) {
    if (!g_qos_initialized) return -1;
    return qos_llm_suggest_command(g_qos_sched.kernel_llm, context, suggestion, 64);
}
