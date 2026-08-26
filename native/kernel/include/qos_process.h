#ifndef QOS_PROCESS_H
#define QOS_PROCESS_H

#include <stdint.h>
#include <stddef.h>
#include "triad_kernel_rt.h"

#define QOS_MAX_PROCESSES 64
#define QOS_MAX_COUPLINGS 32
#define QOS_MAX_GRID 1024

typedef struct {
    int src_pid;
    int dst_pid;
    double kappa;
    double duration;
    int active;
} QosCoupling;

typedef struct {
    int pid;
    char name[64];
    
    // P1: Wavefunction (não colapsa, evolui)
    TriadCplx *psi;
    int N;
    int D;
    double L;
    double dx;
    
    // P2: Memory field (correlações temporais)
    double *y;
    int M_y;
    
    // P3: Reservoir dynamics
    double Lambda;
    double alpha;
    double Gamma;
    double sigma;
    double f_FDT;
    
    // Auto-organização
    double energy;
    double crystallinity;
    double k_star;
    double ipr;
    
    // Estado do processo
    int state;
    uint64_t created_at;
    uint64_t last_evolution;
    
    // Acoplamentos (comunicação entre processos)
    QosCoupling couplings[QOS_MAX_COUPLINGS];
    int n_couplings;
    
} QosProcess;

typedef struct {
    double global_time;
    double dt;
    int n_processes;
    QosProcess processes[QOS_MAX_PROCESSES];
    
    // Campo global (todos os processos compartilham)
    TriadCplx *global_field;
    double *global_potential;
    
    // LLM interno (opcional, pode ser NULL)
    void *llm_context;
    
} QosRuntime;

void qos_init(void);
int qos_create_process(const char *name, int N, int D, double L);
void qos_destroy_process(int pid);

void qos_set_parameters(int pid, double Lambda, double alpha, double Gamma);
void qos_couple(int src_pid, int dst_pid, double kappa);
void qos_decouple(int src_pid, int dst_pid);

void qos_evolve_all(double T);
void qos_evolve_process(int pid, double dt);

double qos_measure_energy(int pid);
double qos_measure_crystallinity(int pid);
void qos_get_state(int pid, TriadCplx *out_psi, double *out_y);

int qos_is_crystallized(int pid);
void qos_wait_equilibrium(int pid, double tolerance);

#endif