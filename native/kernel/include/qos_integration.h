#ifndef QOS_INTEGRATION_H
#define QOS_INTEGRATION_H

#include "triad_kernel.h"
#include "qos_process.h"
#include "qos_llm.h"

typedef struct {
    TriadProc base;
    QosProcess *quantum;
    QosLLM *llm_ctx;
    char source_file[256];
    int crystallized;
    double crystallinity;
} QosProc;

typedef struct {
    TriadThread base;
    int substrate_id;
    double evolution_time;
    double last_energy;
} QosThread;

typedef struct {
    TriadProc classic_procs[MAX_PROCS];
    int n_classic;
    QosProc quantum_procs[MAX_PROCS];
    int n_quantum;
    QosRuntime *qos_rt;
    QosLLM *kernel_llm;
    double global_time;
    double dt;
    int quantum_enabled;
    int llm_enabled;
} QosScheduler;

void qos_scheduler_init(void);
void qos_kernel_init(void);
int qos_proc_create(const char *name, int is_quantum);
int qos_proc_from_tri_file(const char *path);
void qos_sched_tick(void);
void qos_sched_yield(void);
int qos_sched_get_current(void);
int64_t sys_quantum_create(uint64_t N, uint64_t L, uint64_t Lambda);
int64_t sys_quantum_evolve(uint64_t pid, uint64_t T);
int64_t sys_quantum_measure(uint64_t pid, uint64_t observable);
int64_t sys_quantum_couple(uint64_t src, uint64_t dst, uint64_t kappa);
int64_t sys_llm_generate(uint64_t prompt_ptr, uint64_t out_ptr, uint64_t max_len);
int64_t sys_llm_absorb(uint64_t data_ptr, uint64_t len);
int qos_vfs_save_quantum_state(const char *path, int pid);
int qos_vfs_load_quantum_state(const char *path, int pid);
void qos_shell_run(void);
int qos_shell_execute(const char *cmd);
int qos_eval_tri_code(const char *code, char *output, int max_len);
int qos_run_tri_file(const char *path);
void qos_llm_absorb_process(int pid);
void qos_llm_absorb_text(const char *text);
int qos_llm_suggest(const char *context, char *suggestion);

#endif
