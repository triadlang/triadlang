// Triad Quantum Solver - Kernel Version
// Motor P1 + P2 + P3 para o kernel do OS
// Adaptado de native/c/src/triad_solver.c e triad_multi_runtime.c

#include "triad_kernel_rt.h"
#include "qos_process.h"
#include <stddef.h>

// FFT in-place para o kernel
static void fft_kernel(TriadCplx *data, int N, int sign) {
    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < N - 1; i++) {
        if (i < j) {
            TriadCplx tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
        int k = N >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
    
    // Cooley-Tukey FFT
    for (int L = 2; L <= N; L <<= 1) {
        double theta = sign * 2.0 * M_PI / L;
        double w_re = triad_cos(theta);
        double w_im = triad_sin(theta);
        
        for (int i = 0; i < N; i += L) {
            double cur_re = 1.0;
            double cur_im = 0.0;
            
            for (int k = 0; k < L / 2; k++) {
                int idx1 = i + k;
                int idx2 = i + k + L / 2;
                
                double u_re = data[idx1].re;
                double u_im = data[idx1].im;
                double v_re = data[idx2].re * cur_re - data[idx2].im * cur_im;
                double v_im = data[idx2].re * cur_im + data[idx2].im * cur_re;
                
                data[idx1].re = u_re + v_re;
                data[idx1].im = u_im + v_im;
                data[idx2].re = u_re - v_re;
                data[idx2].im = u_im - v_im;
                
                double next_re = cur_re * w_re - cur_im * w_im;
                double next_im = cur_re * w_im + cur_im * w_re;
                cur_re = next_re;
                cur_im = next_im;
            }
        }
    }
    
    // Normalize
    if (sign == -1) {
        double invN = 1.0 / N;
        for (int i = 0; i < N; i++) {
            data[i].re *= invN;
            data[i].im *= invN;
        }
    }
}

// Evoluir o campo ψ por um timestep (P1: Schrödinger evolution)
static void evolve_psi(QosProcess *p, double dt) {
    if (!p->psi || p->N <= 0) return;
    
    int N = p->N;
    double L = p->L;
    double dx = L / N;
    
    // FFT para espaço de frequência
    fft_kernel(p->psi, N, -1);
    
    // Aplicar fase cinética (kinetic term in Fourier space)
    double dk = 2.0 * M_PI / L;
    for (int k = 0; k < N; k++) {
        double kx = (k <= N/2) ? k * dk : (k - N) * dk;
        double phase = -0.5 * kx * kx * dt;
        double c = triad_cos(phase);
        double s = triad_sin(phase);
        
        double re = p->psi[k].re * c - p->psi[k].im * s;
        double im = p->psi[k].re * s + p->psi[k].im * c;
        p->psi[k].re = re;
        p->psi[k].im = im;
    }
    
    // FFT inversa
    fft_kernel(p->psi, N, 1);
    
    // Aplicar potencial e não-linearidade (P1 + P2 + P3 integrados!)
    double Lambda = p->Lambda;
    double alpha = p->alpha;
    double Gamma = p->Gamma;
    
    for (int i = 0; i < N; i++) {
        double rho = p->psi[i].re * p->psi[i].re + p->psi[i].im * p->psi[i].im;
        
        // P1: Nonlinear self-interaction
        double V_nonlin = Lambda * rho + alpha * rho * rho;
        
        // P2: Memory field contribution
        double mem_term = 0.0;
        if (p->y && i < p->M_y) {
            mem_term = p->y[i];
        }
        
        // P3: Reservoir dynamics (noise + dissipation)
        // (usamos ruído determinístico baseado no estado)
        uint64_t state = (uint64_t)(i + (int)(p->energy * 1000)) ^ 0xDEADBEEF;
        double noise = triad_randn(&state) * 0.001;
        
        // Potencial total
        double V_total = V_nonlin + mem_term;
        
        // Aplicar fase do potencial
        double angle = -V_total * dt;
        double c = triad_cos(angle);
        double s = triad_sin(angle);
        
        double re = p->psi[i].re * c - p->psi[i].im * s;
        double im = p->psi[i].re * s + p->psi[i].im * c;
        p->psi[i].re = re + noise * Gamma;
        p->psi[i].im = im + noise * Gamma;
        
        // Evoluir memória (P2)
        if (p->y && i < p->M_y) {
            // Memory decay + accumulation
            p->y[i] = p->y[i] * (1.0 - Gamma * dt) + rho * dt * 0.1;
        }
    }
}

// Calcular observáveis (energia, cristalinidade)
static void compute_observables(QosProcess *p) {
    if (!p->psi || p->N <= 0) return;
    
    int N = p->N;
    double L = p->L;
    double dx = L / N;
    
    // Norma
    double norm = 0.0;
    for (int i = 0; i < N; i++) {
        double rho = p->psi[i].re * p->psi[i].re + p->psi[i].im * p->psi[i].im;
        norm += rho;
    }
    norm *= dx;
    
    // Energia cinética
    double kinetic = 0.0;
    for (int i = 1; i < N; i++) {
        double dpsi_re = (p->psi[i].re - p->psi[i-1].re) / dx;
        double dpsi_im = (p->psi[i].im - p->psi[i-1].im) / dx;
        double dpsi2 = dpsi_re * dpsi_re + dpsi_im * dpsi_im;
        kinetic += dpsi2;
    }
    kinetic *= 0.5 * dx;
    
    // Energia potencial (não-linear)
    double potential = 0.0;
    for (int i = 0; i < N; i++) {
        double rho = p->psi[i].re * p->psi[i].re + p->psi[i].im * p->psi[i].im;
        potential += 0.5 * p->Lambda * rho * rho;
    }
    potential *= dx;
    
    p->energy = kinetic + potential;
    
    // Cristalinidade (medida de ordem)
    double grad_norm = 0.0;
    for (int i = 1; i < N; i++) {
        double dpsi_re = p->psi[i].re - p->psi[i-1].re;
        double dpsi_im = p->psi[i].im - p->psi[i-1].im;
        grad_norm += dpsi_re * dpsi_re + dpsi_im * dpsi_im;
    }
    
    // Cristalinidade alta quando gradiente é baixo (estrutura)
    p->crystallinity = 1.0 / (1.0 + grad_norm * dx);
    
    // k_star (correlação de Fourier)
    double k_star = 0.0;
    int mid = N / 2;
    for (int k = mid / 2; k < mid; k++) {
        double rho = p->psi[k].re * p->psi[k].re + p->psi[k].im * p->psi[k].im;
        k_star += rho;
    }
    p->k_star = k_star / mid;
}

// Transferir densidade entre substratos acoplados (comunicação)
static void transfer_density(QosProcess *src, QosProcess *dst, double kappa) {
    if (!src->psi || !dst->psi) return;
    
    int minN = src->N < dst->N ? src->N : dst->N;
    
    // Transferência de densidade ponderada por kappa
    for (int i = 0; i < minN; i++) {
        double rho_src = src->psi[i].re * src->psi[i].re + src->psi[i].im * src->psi[i].im;
        double phase_src = triad_atan2(src->psi[i].im, src->psi[i].re);
        
        // Perturbação no destino (comunicação via campo)
        double perturbation = kappa * rho_src * 0.01;
        
        // Aplicar no destino com fase preservada
        double rho_dst = dst->psi[i].re * dst->psi[i].re + dst->psi[i].im * dst->psi[i].im;
        double phase_dst = triad_atan2(dst->psi[i].im, dst->psi[i].re);
        
        // Amplitude aumenta levemente
        double new_rho = rho_dst + perturbation;
        if (new_rho > 0) {
            double amp = triad_sqrt(new_rho);
            dst->psi[i].re = amp * triad_cos(phase_dst);
            dst->psi[i].im = amp * triad_sin(phase_dst);
        }
    }
}

// Função atan2 simples para o kernel
double triad_atan2(double y, double x) {
    if (x > 0) return triad_atan(y / x);
    if (x < 0) {
        if (y >= 0) return triad_atan(y / x) + M_PI;
        return triad_atan(y / x) - M_PI;
    }
    if (y > 0) return M_PI / 2;
    if (y < 0) return -M_PI / 2;
    return 0;
}

// atan approximation
double triad_atan(double x) {
    double a = triad_fabs(x);
    double s = x < 0 ? -1 : 1;
    
    // Approximation for |x| <= 1
    if (a <= 1) {
        return s * (M_PI / 4 * a - a * (a - 1) * (0.2447 + 0.0663 * a));
    }
    
    // For |x| > 1, use atan(x) = pi/2 - atan(1/x)
    return s * (M_PI / 2 - triad_atan(1.0 / a));
}

// Inicializar processo quântico
void qos_kernel_init_process(QosProcess *p, int N, double L) {
    if (!p) return;
    
    p->N = N;
    p->D = 1;
    p->L = L;
    p->dx = L / N;
    
    // Alocar campo ψ
    p->psi = (TriadCplx *)triad_kmalloc(N * sizeof(TriadCplx));
    if (!p->psi) return;
    
    // Inicializar com estado fundamental (gaussiana)
    double sigma = L / 10.0;
    double norm = 0;
    for (int i = 0; i < N; i++) {
        double x = (i - N/2) * p->dx;
        p->psi[i].re = triad_exp(-x * x / (2 * sigma * sigma));
        p->psi[i].im = 0;
        norm += p->psi[i].re * p->psi[i].re;
    }
    
    // Normalizar
    norm = triad_sqrt(norm);
    if (norm > 1e-15) {
        for (int i = 0; i < N; i++) {
            p->psi[i].re /= norm;
        }
    }
    
    // Alocar memória P2
    p->M_y = N;
    p->y = (double *)triad_kmalloc(N * sizeof(double));
    if (p->y) {
        triad_memset(p->y, 0, N * sizeof(double));
    }
    
    // Parâmetros P3 padrão
    p->Lambda = 4.0;    // Não-linearidade
    p->alpha = 0.1;     // Correção
    p->Gamma = 0.01;    // Dissipação
    p->sigma = 0.1;     // Ruído
    p->f_FDT = 1.0;     // Fluctuation-dissipation
    
    // Estado inicial: caótico
    p->energy = 0;
    p->crystallinity = 0;
    p->k_star = 0;
    p->state = 0;
}

// Evoluir todos os processos quânticos
void qos_kernel_evolve_all(QosRuntime *rt, double T) {
    if (!rt) return;
    
    int n_steps = (int)(T / 0.001);  // dt fixo de 0.001
    if (n_steps < 1) n_steps = 1;
    
    for (int step = 0; step < n_steps; step++) {
        // Para cada processo
        for (int i = 0; i < rt->n_processes; i++) {
            QosProcess *p = &rt->processes[i];
            
            // Evoluir P1 + P2 + P3
            evolve_psi(p, 0.001);
            
            // Calcular observáveis
            compute_observables(p);
            
            // Detectar cristalização
            if (p->crystallinity > 0.8 && p->state == 0) {
                p->state = 1;  // Equilibrado
            }
            if (p->crystallinity > 0.95 && p->state == 1) {
                p->state = 2;  // Cristalizado
            }
        }
        
        // Processar acoplamentos (comunicação entre processos)
        for (int i = 0; i < rt->n_processes; i++) {
            QosProcess *src = &rt->processes[i];
            
            for (int c = 0; c < src->n_couplings; c++) {
                if (!src->couplings[c].active) continue;
                
                int dst_idx = src->couplings[c].dst_pid;
                if (dst_idx < 0 || dst_idx >= rt->n_processes) continue;
                
                QosProcess *dst = &rt->processes[dst_idx];
                double kappa = src->couplings[c].kappa;
                
                // Transferir densidade (comunicação)
                transfer_density(src, dst, kappa);
            }
        }
        
        rt->global_time += 0.001;
    }
}