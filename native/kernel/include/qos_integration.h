#ifndef QOS_INTEGRATION_H
#define QOS_INTEGRATION_H

#include "triad_kernel.h"
#include "qos_process.h"
#include "qos_llm.h"

// Integração do QuantumOS com o kernel TriadOS existente
// O OS parece normal por fora, mas funciona de forma diferente por dentro

// Mapeamento: Processo tradicional -> Substrato quântico
// 
// TriadProc (tradicional)     QosProcess (quântico)
// ---------------------       ---------------------
// PID                         pid
// stack/heap                  ψ (campo)
// context                     y (memória)
// scheduling                   evolução temporal
// IPC (pipes)                 acoplamento κ
// memory pages                amplitudes do campo

// Estende TriadProc para incluir estado quântico
typedef struct {
    TriadProc base;           // Estrutura tradicional
    
    // Estado quântico (camada interna, invisível para o usuário)
    QosProcess *quantum;      // NULL = processo clássico, não-NULL = quântico
    
    // LLM context para este processo
    QosLLM *llm_ctx;
    
    // Arquivo .tri fonte (se houver)
    char source_file[256];
    
    // Estado de cristalização
    int crystallized;         // 0 = caótico, 1 = equilibrado, 2 = cristalizado
    double crystallinity;     // 0.0 - 1.0
    
} QosProc;

// Estende TriadThread para incluir evolução quântica
typedef struct {
    TriadThread base;         // Thread tradicional
    
    // Quantum evolution state
    int substrate_id;         // ID do substrato quântico associado
    double evolution_time;    // Tempo de evolução acumulado
    double last_energy;       // Última energia medida
    
} QosThread;

// Scheduler híbrido: round-robin clássico + evolução quântica
typedef struct {
    // Processos clássicos
    TriadProc classic_procs[MAX_PROCS];
    int n_classic;
    
    // Processos quânticos
    QosProc quantum_procs[MAX_PROCS];
    int n_quantum;
    
    // Runtime quântico global
    QosRuntime *qos_rt;
    
    // LLM global do kernel
    QosLLM *kernel_llm;
    
    // Tempo global (evolução do campo)
    double global_time;
    double dt;                // Time step
    
    // Flags
    int quantum_enabled;       // 1 = usar evolução quântica
    int llm_enabled;           // 1 = LLM ativo
    
} QosScheduler;

// Inicialização
void qos_scheduler_init(void);
void qos_kernel_init(void);

// Criação de processos
int qos_proc_create(const char *name, int is_quantum);
int qos_proc_from_tri_file(const char *path);

// Scheduler híbrido
void qos_sched_tick(void);
void qos_sched_yield(void);
int qos_sched_get_current(void);

// Syscalls quânticas (estendem syscalls tradicionais)
int64_t sys_quantum_create(uint64_t N, uint64_t L, uint64_t Lambda);
int64_t sys_quantum_evolve(uint64_t pid, uint64_t T);
int64_t sys_quantum_measure(uint64_t pid, uint64_t observable);
int64_t sys_quantum_couple(uint64_t src, uint64_t dst, uint64_t kappa);
int64_t sys_llm_generate(uint64_t prompt_ptr, uint64_t out_ptr, uint64_t max_len);
int64_t sys_llm_absorb(uint64_t data_ptr, uint64_t len);

// VFS estendido para armazenar estados quânticos
int qos_vfs_save_quantum_state(const char *path, int pid);
int qos_vfs_load_quantum_state(const char *path, int pid);

// Shell integrado
void qos_shell_run(void);
int qos_shell_execute(const char *cmd);

// Integração com interpretador TriadLang existente
int qos_eval_tri_code(const char *code, char *output, int max_len);
int qos_run_tri_file(const char *path);

// LLM integration
void qos_llm_absorb_process(int pid);
void qos_llm_absorb_text(const char *text);
int qos_llm_suggest(const char *context, char *suggestion);

#endif