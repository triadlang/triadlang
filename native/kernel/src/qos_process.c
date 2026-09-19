#include "qos_process.h"
#include "triad_rt.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static QosRuntime g_qos;
static int g_qos_initialized = 0;

static uint64_t g_time_ns = 0;

static double randn(void) {
    static uint64_t state = 0xDEADBEEF12345678ULL;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    
    double u1 = (double)(state >> 11) / (double)(1ULL << 53);
    double u2 = (double)(state >> 33) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void fft_1d(TriadCplx *in, TriadCplx *out, int N, int sign) {
    for (int k = 0; k < N; k++) {
        out[k].re = 0;
        out[k].im = 0;
        for (int n = 0; n < N; n++) {
            double angle = sign * 2.0 * M_PI * k * n / N;
            double c = cos(angle);
            double s = sin(angle);
            out[k].re += in[n].re * c - in[n].im * s;
            out[k].im += in[n].re * s + in[n].im * c;
        }
        if (sign > 0) {
            out[k].re /= N;
            out[k].im /= N;
        }
    }
}

static void evolve_step(QosProcess *p, double dt, double V_ext, double V_coupled) {
    if (!p->psi || p->N <= 0) return;
    
    TriadCplx *psi_freq = (TriadCplx *)malloc(p->N * sizeof(TriadCplx));
    if (!psi_freq) return;
    fft_1d(p->psi, psi_freq, p->N, -1);
    double dk = 2.0 * M_PI / p->L;
    for (int k = 0; k < p->N; k++) {
        double kx = (k < p->N/2) ? k * dk : (k - p->N) * dk;
        double phase = -0.5 * kx * kx * dt;
        double c = cos(phase);
        double s = sin(phase);
        double re = psi_freq[k].re * c - psi_freq[k].im * s;
        double im = psi_freq[k].re * s + psi_freq[k].im * c;
        psi_freq[k].re = re;
        psi_freq[k].im = im;
    }
    
    fft_1d(psi_freq, p->psi, p->N, 1);
    
    double Lambda = p->Lambda;
    double alpha = p->alpha;
    double Gamma = p->Gamma;
    double V_total = V_ext + V_coupled;
    for (int i = 0; i < p->N; i++) {
        double rho = p->psi[i].re * p->psi[i].re + p->psi[i].im * p->psi[i].im;
        double V_nonlinear = Lambda * rho + alpha * rho * rho;
        double memory_term = (p->y && i < p->M_y) ? p->y[i] : 0.0;
        double reservoir_noise = Gamma > 0 ? randn() * sqrt(Gamma) : 0;
        double V_eff = V_total + V_nonlinear + memory_term;
        double angle = -V_eff * dt;
        double c = cos(angle);
        double s = sin(angle);
        double re = p->psi[i].re * c - p->psi[i].im * s;
        double im = p->psi[i].re * s + p->psi[i].im * c;
        p->psi[i].re = re;
        p->psi[i].im = im;
        
        if (p->y && i < p->M_y) {
            p->y[i] = p->y[i] * 0.99 + 0.01 * rho + reservoir_noise * 0.001;
        }
    }
    
    free(psi_freq);
}

static void compute_observables(QosProcess *p) {
    if (!p->psi || p->N <= 0) return;
    
    double norm = 0;
    double mean_x = 0;
    double mean_x2 = 0;
    
    for (int i = 0; i < p->N; i++) {
        double rho = p->psi[i].re * p->psi[i].re + p->psi[i].im * p->psi[i].im;
        norm += rho;
        mean_x += i * rho;
        mean_x2 += i * i * rho;
    }
    
    if (norm > 1e-15) {
        mean_x /= norm;
        mean_x2 /= norm;
        p->energy = mean_x2 - mean_x * mean_x;
    }
    double crystallinity = 0;
    for (int i = 1; i < p->N; i++) {
        double dpsi_re = p->psi[i].re - p->psi[i-1].re;
        double dpsi_im = p->psi[i].im - p->psi[i-1].im;
        crystallinity += sqrt(dpsi_re*dpsi_re + dpsi_im*dpsi_im);
    }
    p->crystallinity = 1.0 / (1.0 + crystallinity / p->N);
    
    double k_star = 0;
    for (int i = 0; i < p->N/2; i++) {
        double re = p->psi[i].re * p->psi[p->N-1-i].re - p->psi[i].im * p->psi[p->N-1-i].im;
        double im = p->psi[i].re * p->psi[p->N-1-i].im + p->psi[i].im * p->psi[p->N-1-i].re;
        k_star += sqrt(re*re + im*im);
    }
    p->k_star = k_star / (p->N / 2);
}

static void compute_couplings(double *V_out, int n, QosProcess *processes) {
    for (int i = 0; i < n; i++) {
        V_out[i] = 0;
    }
    
    for (int i = 0; i < n; i++) {
        QosProcess *src = &processes[i];
        
        for (int c = 0; c < src->n_couplings; c++) {
            if (!src->couplings[c].active) continue;
            
            int dst_idx = src->couplings[c].dst_pid;
            if (dst_idx < 0 || dst_idx >= n) continue;
            
            QosProcess *dst = &processes[dst_idx];
            double kappa = src->couplings[c].kappa;
            
            int minN = src->N < dst->N ? src->N : dst->N;
            for (int k = 0; k < minN; k++) {
                double rho_src = src->psi[k].re * src->psi[k].re + src->psi[k].im * src->psi[k].im;
                V_out[dst_idx * QOS_MAX_GRID + k] += kappa * rho_src;
            }
        }
    }
}

void qos_init(void) {
    if (g_qos_initialized) return;
    
    memset(&g_qos, 0, sizeof(g_qos));
    g_qos.dt = 0.001;
    g_qos.global_time = 0;
    g_qos.n_processes = 0;
    g_qos.global_field = NULL;
    g_qos.global_potential = NULL;
    g_qos.llm_context = NULL;
    
    g_qos_initialized = 1;
}

int qos_create_process(const char *name, int N, int D, double L) {
    if (!g_qos_initialized) qos_init();
    
    if (g_qos.n_processes >= QOS_MAX_PROCESSES) return -1;
    
    int pid = g_qos.n_processes;
    QosProcess *p = &g_qos.processes[pid];
    
    memset(p, 0, sizeof(QosProcess));
    p->pid = pid;
    snprintf(p->name, sizeof(p->name), "%s", name);
    
    p->N = N;
    p->D = D;
    p->L = L;
    p->dx = L / N;
    p->psi = (TriadCplx *)malloc(N * sizeof(TriadCplx));
    if (p->psi) {
        memset(p->psi, 0, N * sizeof(TriadCplx));
        double x0 = 0;
        double sigma = L / 10.0;
        for (int i = 0; i < N; i++) {
            double x = (i - N/2) * p->dx;
            double amp = exp(-(x-x0)*(x-x0) / (2*sigma*sigma));
            p->psi[i].re = amp / sqrt(2.0 * M_PI * sigma * sigma);
            p->psi[i].im = 0;
        }
    }
    
    p->M_y = N;
    p->y = (double *)malloc(N * sizeof(double));
    if (p->y) memset(p->y, 0, N * sizeof(double));
    p->Lambda = 4.0;
    p->alpha = 0.1;
    p->Gamma = 0.01;
    p->sigma = 0.1;
    p->f_FDT = 1.0;
    p->energy = 0;
    p->crystallinity = 0;
    p->k_star = 0;
    p->state = 0;
    p->created_at = g_time_ns;
    p->last_evolution = g_time_ns;
    
    g_qos.n_processes++;
    
    return pid;
}

void qos_destroy_process(int pid) {
    if (pid < 0 || pid >= g_qos.n_processes) return;
    
    QosProcess *p = &g_qos.processes[pid];
    if (p->psi) free(p->psi);
    if (p->y) free(p->y);
    memset(p, 0, sizeof(QosProcess));
}

void qos_set_parameters(int pid, double Lambda, double alpha, double Gamma) {
    if (pid < 0 || pid >= g_qos.n_processes) return;
    
    QosProcess *p = &g_qos.processes[pid];
    p->Lambda = Lambda;
    p->alpha = alpha;
    p->Gamma = Gamma;
}

void qos_couple(int src_pid, int dst_pid, double kappa) {
    if (src_pid < 0 || src_pid >= g_qos.n_processes) return;
    if (dst_pid < 0 || dst_pid >= g_qos.n_processes) return;
    
    QosProcess *src = &g_qos.processes[src_pid];
    
    for (int i = 0; i < src->n_couplings; i++) {
        if (src->couplings[i].dst_pid == dst_pid) {
            src->couplings[i].kappa = kappa;
            src->couplings[i].active = 1;
            return;
        }
    }
    
    if (src->n_couplings >= QOS_MAX_COUPLINGS) return;
    
    src->couplings[src->n_couplings].src_pid = src_pid;
    src->couplings[src->n_couplings].dst_pid = dst_pid;
    src->couplings[src->n_couplings].kappa = kappa;
    src->couplings[src->n_couplings].duration = 0;
    src->couplings[src->n_couplings].active = 1;
    src->n_couplings++;
}

void qos_decouple(int src_pid, int dst_pid) {
    if (src_pid < 0 || src_pid >= g_qos.n_processes) return;
    
    QosProcess *src = &g_qos.processes[src_pid];
    for (int i = 0; i < src->n_couplings; i++) {
        if (src->couplings[i].dst_pid == dst_pid) {
            src->couplings[i].active = 0;
        }
    }
}

void qos_evolve_process(int pid, double dt) {
    if (pid < 0 || pid >= g_qos.n_processes) return;
    
    QosProcess *p = &g_qos.processes[pid];
    
    double V_ext = 0;
    for (int i = 0; i < p->N; i++) {
        V_ext += 0.5 * (i - p->N/2) * (i - p->N/2) * p->dx * p->dx;
    }
    V_ext /= p->N;
    
    evolve_step(p, dt, V_ext, 0);
    compute_observables(p);
    
    p->last_evolution = g_time_ns;
    
    if (p->crystallinity > 0.8 && p->state == 0) {
        p->state = 1;
    }
    if (p->crystallinity > 0.95 && p->state == 1) {
        p->state = 2;
    }
}

void qos_evolve_all(double T) {
    if (!g_qos_initialized) qos_init();
    
    int n_steps = (int)(T / g_qos.dt);
    
    double *V_coupled = (double *)malloc(g_qos.n_processes * QOS_MAX_GRID * sizeof(double));
    if (!V_coupled) return;
    for (int step = 0; step < n_steps; step++) {
        compute_couplings(V_coupled, g_qos.n_processes, g_qos.processes);
        for (int i = 0; i < g_qos.n_processes; i++) {
            QosProcess *p = &g_qos.processes[i];
            
            double V_ext = 0;
            for (int k = 0; k < p->N; k++) {
                V_ext += 0.5 * (k - p->N/2) * (k - p->N/2) * p->dx * p->dx;
            }
            V_ext /= p->N;
            
            evolve_step(p, g_qos.dt, V_ext, V_coupled[i * QOS_MAX_GRID]);
            compute_observables(p);
            
            if (p->crystallinity > 0.8 && p->state == 0) p->state = 1;
            if (p->crystallinity > 0.95 && p->state == 1) p->state = 2;
        }
        
        g_qos.global_time += g_qos.dt;
        g_time_ns++;
    }
    
    free(V_coupled);
}

double qos_measure_energy(int pid) {
    if (pid < 0 || pid >= g_qos.n_processes) return 0;
    return g_qos.processes[pid].energy;
}

double qos_measure_crystallinity(int pid) {
    if (pid < 0 || pid >= g_qos.n_processes) return 0;
    return g_qos.processes[pid].crystallinity;
}

void qos_get_state(int pid, TriadCplx *out_psi, double *out_y) {
    if (pid < 0 || pid >= g_qos.n_processes) return;
    
    QosProcess *p = &g_qos.processes[pid];
    if (out_psi && p->psi) memcpy(out_psi, p->psi, p->N * sizeof(TriadCplx));
    if (out_y && p->y) memcpy(out_y, p->y, p->M_y * sizeof(double));
}

int qos_is_crystallized(int pid) {
    if (pid < 0 || pid >= g_qos.n_processes) return 0;
    return g_qos.processes[pid].state >= 2;
}

void qos_wait_equilibrium(int pid, double tolerance) {
    if (pid < 0 || pid >= g_qos.n_processes) return;
    
    QosProcess *p = &g_qos.processes[pid];
    double prev_cryst = p->crystallinity;
    int stable_count = 0;
    
    while (stable_count < 10) {
        qos_evolve_process(pid, g_qos.dt * 100);
        
        double diff = fabs(p->crystallinity - prev_cryst);
        if (diff < tolerance) {
            stable_count++;
        } else {
            stable_count = 0;
        }
        prev_cryst = p->crystallinity;
    }
}