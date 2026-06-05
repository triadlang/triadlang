/*
 * triad_multi_runtime.c — Native C port of runtime/multi_runtime.py.
 *
 * Supports 1D, 2D, 3D substrates lockstep with the SAME split-step
 * Strang scheme used by triad_solve_1d / triad_solve_2d / triad_solve_3d.
 *
 * P1+P2+P3 PRESERVATION (all dimensions):
 *   P1 (Schrödinger non-linear, Λ|ψ|²) — present in V_total = ... + Λ ρ.
 *   P2 (hierarchical memory) — V_mem = Σ_j λ_j y_j with OU half-steps
 *       around the potential phase, broadcast over the full ND grid.
 *   P3 (fluctuation-dissipation lock) — noise_amp = sqrt(f_FDT dt / dx^D),
 *       added between the closing OU half-step and the closing
 *       half-linear step. f_FDT == 0 ⇒ no noise, but the other pillars
 *       still run.
 *
 * Coupling on top of that: V_couple is added to V_total in every step.
 * Cross-dimensional couplings use triad_mr_project_to_shape to bridge
 * src and dst grids (sum-projection for down-cast, Gaussian envelope
 * for up-cast — same rule as runtime/multi_runtime.py).
 */
#define _POSIX_C_SOURCE 200809L
#include "triad_multi_runtime.h"
#include "triad_rt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* portable strdup */
static char *_dup_str(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *out = (char *)malloc(n);
    if (out) memcpy(out, s, n);
    return out;
}

/* ── helpers ───────────────────────────────────────────────────── */

static TriadCplx _cmul(TriadCplx a, TriadCplx b) {
    return (TriadCplx){ a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re };
}
static double _cabs2(TriadCplx a) { return a.re*a.re + a.im*a.im; }

static uint64_t _xs64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    return x;
}
static double _randn(uint64_t *state) {
    double u1 = (double)(_xs64(state) >> 11) / (double)(1ULL << 53);
    double u2 = (double)(_xs64(state) >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * TRIAD_PI * u2);
}

static int64_t _grid_size(int D, int N) {
    if (D == 1) return (int64_t)N;
    if (D == 2) return (int64_t)N * (int64_t)N;
    if (D == 3) return (int64_t)N * (int64_t)N * (int64_t)N;
    return 0;
}

/* dx^D helper for noise normalisation */
static double _dx_pow_D(double dx, int D) {
    if (D == 1) return dx;
    if (D == 2) return dx * dx;
    if (D == 3) return dx * dx * dx;
    return dx;
}

/* ── V_ext builder ND ──────────────────────────────────────────── */

static void _build_V_ext_static(TriadSubstrate *s) {
    double *V = s->V_ext_static;
    int64_t G = s->grid_size;
    if (!s->V_ext || strcmp(s->V_ext, "none") == 0 || strcmp(s->V_ext, "zero") == 0) {
        memset(V, 0, sizeof(double) * (size_t)G);
        return;
    }
    if (strcmp(s->V_ext, "harmonic") == 0) {
        double c = 0.5 * s->m * s->omega * s->omega;
        int N = s->N;
        if (s->D == 1) {
            for (int i = 0; i < N; ++i) V[i] = c * s->x[i] * s->x[i];
        } else if (s->D == 2) {
            for (int i = 0; i < N; ++i) {
                double xi = s->x[i];
                for (int j = 0; j < N; ++j) {
                    double xj = s->x[j];
                    V[(int64_t)i * N + j] = c * (xi * xi + xj * xj);
                }
            }
        } else {  /* 3D */
            for (int i = 0; i < N; ++i) {
                double xi = s->x[i];
                for (int j = 0; j < N; ++j) {
                    double xj = s->x[j];
                    for (int k = 0; k < N; ++k) {
                        double xk = s->x[k];
                        V[(int64_t)i*N*N + (int64_t)j*N + k]
                            = c * (xi*xi + xj*xj + xk*xk);
                    }
                }
            }
        }
        return;
    }
    /* unknown shape → zeros (first port; templates extend later) */
    memset(V, 0, sizeof(double) * (size_t)G);
}

static void _effective(TriadSubstrate *s) {
    if (s->mode == 0) {
        s->Lambda_e = 0; s->alpha_e = 0; s->Gamma_e = 0; s->f_FDT_e = 0;
        for (int j = 0; j < s->M; ++j) s->lam_e[j] = 0.0;
    } else if (s->mode == 1) {
        s->Lambda_e = 0; s->alpha_e = 0;
        s->Gamma_e = s->Gamma; s->f_FDT_e = s->f_FDT;
        for (int j = 0; j < s->M; ++j) s->lam_e[j] = 0.0;
    } else {
        s->Lambda_e = s->Lambda; s->alpha_e = s->alpha;
        s->Gamma_e = s->Gamma;   s->f_FDT_e = s->f_FDT;
        memcpy(s->lam_e, s->lam, sizeof(double) * (size_t)s->M);
    }
}

/* Build half_lin over the ND grid — same fórmula as triad_solve_{1,2,3}d. */
static void _build_propagators(TriadSubstrate *s) {
    int N = s->N;
    double step = s->L / (double)N;
    s->dx = step;
    for (int i = 0; i < N; ++i) s->x[i] = -0.5 * s->L + (double)i * step;

    double *kvec = (double *)malloc(sizeof(double) * (size_t)N);
    triad_fftfreq(N, s->dx, kvec);

    double dt = s->dt;
    double Gamma_decay = (-s->Gamma_e / s->hbar) * (dt / 2.0);
    double mag = exp(Gamma_decay);

    if (s->D == 1) {
        for (int i = 0; i < N; ++i) {
            double k = kvec[i];
            double ak = fabs(k);
            double H_lin = s->hbar * s->hbar * k * k / (2.0 * s->m)
                         + s->alpha_e * pow(ak > 0 ? ak : 1e-30, s->sigma);
            double angle = (-H_lin / s->hbar) * (dt / 2.0);
            s->half_lin[i].re = mag * cos(angle);
            s->half_lin[i].im = mag * sin(angle);
        }
    } else if (s->D == 2) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int64_t idx = (int64_t)i * N + j;
                double k2 = kvec[i]*kvec[i] + kvec[j]*kvec[j];
                double ak = sqrt(k2);
                double H_lin = s->hbar * s->hbar * k2 / (2.0 * s->m)
                             + s->alpha_e * pow(ak > 0 ? ak : 1e-30, s->sigma);
                double angle = (-H_lin / s->hbar) * (dt / 2.0);
                s->half_lin[idx].re = mag * cos(angle);
                s->half_lin[idx].im = mag * sin(angle);
            }
        }
    } else { /* 3D */
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    int64_t idx = (int64_t)i*N*N + (int64_t)j*N + k;
                    double k2 = kvec[i]*kvec[i] + kvec[j]*kvec[j] + kvec[k]*kvec[k];
                    double ak = sqrt(k2);
                    double H_lin = s->hbar * s->hbar * k2 / (2.0 * s->m)
                                 + s->alpha_e * pow(ak > 0 ? ak : 1e-30, s->sigma);
                    double angle = (-H_lin / s->hbar) * (dt / 2.0);
                    s->half_lin[idx].re = mag * cos(angle);
                    s->half_lin[idx].im = mag * sin(angle);
                }
            }
        }
    }
    free(kvec);

    for (int j = 0; j < s->M; ++j) {
        s->ou_decay_half[j] = exp(-s->nu[j] * s->dt * 0.5);
    }

    s->noise_amp = (s->f_FDT_e > 0.0)
                 ? sqrt(s->f_FDT_e * s->dt / _dx_pow_D(s->dx, s->D))
                 : 0.0;

    _build_V_ext_static(s);
}

/* ── default Gaussian initial psi ──────────────────────────────── */

static void _default_psi(TriadSubstrate *s) {
    int N = s->N;
    int64_t G = s->grid_size;
    double dxD = _dx_pow_D(s->dx, s->D);
    double sum = 0.0;
    if (s->D == 1) {
        for (int i = 0; i < N; ++i) {
            double g = exp(-s->x[i] * s->x[i] / 8.0);
            s->psi[i].re = g; s->psi[i].im = 0.0;
            sum += g * g;
        }
    } else if (s->D == 2) {
        for (int i = 0; i < N; ++i) {
            double xi = s->x[i];
            for (int j = 0; j < N; ++j) {
                double xj = s->x[j];
                double g = exp(-(xi*xi + xj*xj) / 8.0);
                int64_t idx = (int64_t)i * N + j;
                s->psi[idx].re = g; s->psi[idx].im = 0.0;
                sum += g * g;
            }
        }
    } else { /* 3D */
        for (int i = 0; i < N; ++i) {
            double xi = s->x[i];
            for (int j = 0; j < N; ++j) {
                double xj = s->x[j];
                for (int k = 0; k < N; ++k) {
                    double xk = s->x[k];
                    double g = exp(-(xi*xi + xj*xj + xk*xk) / 8.0);
                    int64_t idx = (int64_t)i*N*N + (int64_t)j*N + k;
                    s->psi[idx].re = g; s->psi[idx].im = 0.0;
                    sum += g * g;
                }
            }
        }
    }
    double inv = 1.0 / sqrt(sum * dxD);
    for (int64_t i = 0; i < G; ++i) s->psi[i].re *= inv;
}

/* ── substrate lifecycle ───────────────────────────────────────── */

static TriadSubstrate *_substrate_new(int id, const char *name, int D,
                                      int N, double L, double dt,
                                      double hbar, double m, double omega,
                                      double Lambda, double alpha,
                                      double sigma, double Gamma, double f_FDT,
                                      int M, const double *nu, const double *lam,
                                      int p_mode, uint64_t seed,
                                      const char *V_ext,
                                      const TriadCplx *psi_init) {
    if (D < 1 || D > 3) return NULL;
    TriadSubstrate *s = (TriadSubstrate *)calloc(1, sizeof(TriadSubstrate));
    s->id = id;
    s->name = name ? _dup_str(name) : _dup_str("");
    s->D = D;
    s->N = N;
    s->grid_size = _grid_size(D, N);
    s->L = L; s->dt = dt;
    s->hbar = hbar; s->m = m; s->omega = omega;
    s->Lambda = Lambda; s->alpha = alpha; s->sigma = sigma;
    s->Gamma = Gamma; s->f_FDT = f_FDT;
    s->M = M; s->mode = p_mode; s->seed = seed;
    s->V_ext = V_ext;

    int M_alloc = M > 0 ? M : 1;
    s->nu  = (double *)calloc((size_t)M_alloc, sizeof(double));
    s->lam = (double *)calloc((size_t)M_alloc, sizeof(double));
    s->lam_e = (double *)calloc((size_t)M_alloc, sizeof(double));
    s->ou_decay_half = (double *)calloc((size_t)M_alloc, sizeof(double));
    for (int j = 0; j < M; ++j) {
        s->nu[j] = nu[j];
        s->lam[j] = lam[j];
    }

    int64_t G = s->grid_size;
    s->x = (double *)calloc((size_t)N, sizeof(double));
    s->V_ext_static = (double *)calloc((size_t)G, sizeof(double));
    s->half_lin = (TriadCplx *)calloc((size_t)G, sizeof(TriadCplx));
    s->psi = (TriadCplx *)calloc((size_t)G, sizeof(TriadCplx));
    int64_t y_size = (int64_t)M * G;
    s->y = (double *)calloc((size_t)(y_size > 0 ? y_size : 1), sizeof(double));
    s->psi_scratch = (TriadCplx *)calloc((size_t)G, sizeof(TriadCplx));
    s->psi_freq    = (TriadCplx *)calloc((size_t)G, sizeof(TriadCplx));
    s->V_couple = (double *)calloc((size_t)G, sizeof(double));
    s->V_mem    = (double *)calloc((size_t)G, sizeof(double));

    _effective(s);
    _build_propagators(s);

    if (psi_init) {
        memcpy(s->psi, psi_init, sizeof(TriadCplx) * (size_t)G);
    } else {
        _default_psi(s);
    }

    s->active = 1;
    s->record_every = 4;
    s->n_records = 0;
    s->cap_records = 0;
    s->density_traj = NULL;
    s->t_traj = NULL;
    return s;
}

static void _substrate_free(TriadSubstrate *s) {
    if (!s) return;
    free(s->name);
    free(s->nu); free(s->lam); free(s->lam_e);
    free(s->ou_decay_half);
    free(s->x); free(s->V_ext_static);
    free(s->half_lin);
    free(s->psi); free(s->y);
    free(s->psi_scratch); free(s->psi_freq);
    free(s->V_couple); free(s->V_mem);
    free(s->density_traj); free(s->t_traj);
    free(s);
}

/* ── runtime lifecycle ─────────────────────────────────────────── */

TriadMultiRuntime *triad_mr_new(double dt, int record_every) {
    TriadMultiRuntime *rt = (TriadMultiRuntime *)calloc(1, sizeof(TriadMultiRuntime));
    rt->dt = dt;
    rt->record_every = record_every > 0 ? record_every : 4;
    rt->diverged_segment = -1;
    rt->diverged_step = -1;
    return rt;
}

void triad_mr_free(TriadMultiRuntime *rt) {
    if (!rt) return;
    for (int i = 0; i < rt->n_substrates; ++i) _substrate_free(rt->substrates[i]);
    free(rt->substrates);
    for (int s = 0; s < rt->n_segments; ++s) {
        free(rt->segments[s].edges);
        free(rt->segments[s].active_ids);
    }
    free(rt->segments);
    free(rt->diverged_name);
    free(rt);
}

int triad_mr_add_substrate(TriadMultiRuntime *rt, const char *name, int D,
                           int N, double L, double hbar, double m,
                           double omega, double Lambda, double alpha,
                           double sigma, double Gamma, double f_FDT,
                           int M, const double *nu, const double *lam,
                           int p_mode, uint64_t seed,
                           const char *V_ext,
                           const TriadCplx *psi_init) {
    if (!rt) return -1;
    if (rt->n_substrates == rt->cap_substrates) {
        int new_cap = rt->cap_substrates ? rt->cap_substrates * 2 : 4;
        rt->substrates = (TriadSubstrate **)realloc(
            rt->substrates, sizeof(TriadSubstrate *) * (size_t)new_cap);
        rt->cap_substrates = new_cap;
    }
    int id = rt->n_substrates++;
    TriadSubstrate *s = _substrate_new(id, name, D,
        N, L, rt->dt, hbar, m, omega, Lambda, alpha, sigma, Gamma, f_FDT,
        M, nu, lam, p_mode, seed, V_ext, psi_init);
    if (!s) { rt->n_substrates--; return -1; }
    s->record_every = rt->record_every;
    rt->substrates[id] = s;
    return id;
}

int triad_mr_add_substrate_1d(TriadMultiRuntime *rt,
                              const char *name,
                              int N, double L, double hbar, double m,
                              double omega, double Lambda, double alpha,
                              double sigma, double Gamma, double f_FDT,
                              int M, const double *nu, const double *lam,
                              int p_mode, uint64_t seed,
                              const char *V_ext,
                              const TriadCplx *psi_init) {
    return triad_mr_add_substrate(rt, name, /*D*/1,
        N, L, hbar, m, omega, Lambda, alpha, sigma, Gamma, f_FDT,
        M, nu, lam, p_mode, seed, V_ext, psi_init);
}

void triad_mr_add_segment(TriadMultiRuntime *rt,
                          double t_start, double t_end,
                          TriadCouplingEdge *edges, int n_edges,
                          int *active_ids, int n_active) {
    if (rt->n_segments == rt->cap_segments) {
        int new_cap = rt->cap_segments ? rt->cap_segments * 2 : 4;
        rt->segments = (TriadSegment *)realloc(
            rt->segments, sizeof(TriadSegment) * (size_t)new_cap);
        rt->cap_segments = new_cap;
    }
    TriadSegment *seg = &rt->segments[rt->n_segments++];
    seg->t_start = t_start;
    seg->t_end = t_end;
    seg->edges = edges;
    seg->n_edges = n_edges;
    seg->active_ids = active_ids;
    seg->n_active = n_active;
}

TriadSubstrate *triad_mr_get(TriadMultiRuntime *rt, int sid) {
    if (sid < 0 || sid >= rt->n_substrates) return NULL;
    return rt->substrates[sid];
}

TriadSubstrate *triad_mr_find(TriadMultiRuntime *rt, const char *name) {
    for (int i = 0; i < rt->n_substrates; ++i) {
        if (strcmp(rt->substrates[i]->name, name) == 0) return rt->substrates[i];
    }
    return NULL;
}

/* ── _project_to_shape — byte-equivalent of Python rule ────────── */

double *triad_mr_project_to_shape(const double *src_rho,
                                  int src_D, int src_N,
                                  int dst_D, int dst_N) {
    if (!src_rho) return NULL;
    int64_t dst_size = _grid_size(dst_D, dst_N);
    double *out = (double *)calloc((size_t)dst_size, sizeof(double));

    if (src_D == dst_D && src_N == dst_N) {
        int64_t src_size = _grid_size(src_D, src_N);
        memcpy(out, src_rho, sizeof(double) * (size_t)src_size);
        return out;
    }

    /* down-projection */
    if (src_D == 3 && dst_D == 1 && src_N == dst_N) {
        int N = src_N;
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int j = 0; j < N; ++j)
                for (int k = 0; k < N; ++k)
                    s += src_rho[(int64_t)i*N*N + (int64_t)j*N + k];
            out[i] = s;
        }
        return out;
    }
    if (src_D == 2 && dst_D == 1 && src_N == dst_N) {
        int N = src_N;
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int j = 0; j < N; ++j) s += src_rho[(int64_t)i*N + j];
            out[i] = s;
        }
        return out;
    }
    if (src_D == 3 && dst_D == 2 && src_N == dst_N) {
        /* Python only specifies 3D↔1D and 2D↔1D explicitly; 3D→2D falls
         * into the "unsupported" branch returning zeros. Match that. */
        return out;
    }

    /* up-projection: 1D → 2D or 3D via outer product with normalised
     * Gaussian envelope, width = N/8 (Python).
     *   env[k] = exp(-(k - N/2)² / (2 (N/8)²))
     *   env /= env.sum()
     * 1D→3D: proj_2d = src[:, None] * env[None, :]
     *        proj_3d = proj_2d[:, :, None] * env[None, None, :]
     */
    if (src_D == 1 && src_N == dst_N) {
        int N = src_N;
        double *env = (double *)malloc(sizeof(double) * (size_t)N);
        double esum = 0.0;
        double w = (double)N / 8.0;
        double denom = 2.0 * w * w;
        for (int k = 0; k < N; ++k) {
            double d = (double)k - (double)(N / 2);
            env[k] = exp(-(d * d) / denom);
            esum += env[k];
        }
        for (int k = 0; k < N; ++k) env[k] /= esum;

        if (dst_D == 2) {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    out[(int64_t)i*N + j] = src_rho[i] * env[j];
        } else if (dst_D == 3) {
            for (int i = 0; i < N; ++i) {
                double si = src_rho[i];
                for (int j = 0; j < N; ++j) {
                    double sij = si * env[j];
                    for (int k = 0; k < N; ++k)
                        out[(int64_t)i*N*N + (int64_t)j*N + k] = sij * env[k];
                }
            }
        }
        free(env);
        return out;
    }

    /* unsupported combination → zeros (matches Python fallback) */
    return out;
}

/* ── per-step kernel — ND ──────────────────────────────────────── */

/* Apply ND FFT in-place via psi_scratch. */
static void _ndfft(const TriadSubstrate *s, const TriadCplx *in, TriadCplx *out) {
    if (s->D == 1) triad_fft_fn(s->N, in, out);
    else if (s->D == 2) triad_fft2_fn(s->N, in, out);
    else triad_fft3_fn(s->N, in, out);
}
static void _ndifft(const TriadSubstrate *s, const TriadCplx *in, TriadCplx *out) {
    if (s->D == 1) triad_ifft_fn(s->N, in, out);
    else if (s->D == 2) triad_ifft2_fn(s->N, in, out);
    else triad_ifft3_fn(s->N, in, out);
}

static void _step_one(TriadSubstrate *s,
                      double *const *rho_snapshots,
                      TriadSubstrate **all_subs, /* live ptrs for phase_coherent */
                      int n_all_subs,
                      const int *rho_D,    /* per-substrate dimensionality */
                      const int *rho_N,    /* per-substrate axis size */
                      const TriadCouplingEdge *inbound, int n_inbound,
                      uint64_t *rng_state) {
    int64_t G = s->grid_size;
    int M = s->M;
    double dt = s->dt;
    double dt_over_hbar = dt / s->hbar;

    /* 1. half-linear: psi <- ifft(fft(psi) * half_lin) */
    _ndfft(s, s->psi, s->psi_freq);
    for (int64_t i = 0; i < G; ++i)
        s->psi_freq[i] = _cmul(s->psi_freq[i], s->half_lin[i]);
    _ndifft(s, s->psi_freq, s->psi);

    /* 2. rho = |psi|² */
    double *rho = (double *)malloc(sizeof(double) * (size_t)G);
    for (int64_t i = 0; i < G; ++i) rho[i] = _cabs2(s->psi[i]);

    /* 3. OU half-step (broadcast over the full ND grid) */
    if (M > 0) {
        for (int j = 0; j < M; ++j) {
            double od = s->ou_decay_half[j];
            double od1 = 1.0 - od;
            double *yj = &s->y[(int64_t)j * G];
            for (int64_t i = 0; i < G; ++i) yj[i] = od * yj[i] + od1 * rho[i];
        }
    }

    /* 4. V_mem = Σ λ_j y_j */
    for (int64_t i = 0; i < G; ++i) s->V_mem[i] = 0.0;
    if (M > 0) {
        for (int j = 0; j < M; ++j) {
            double lj = s->lam_e[j];
            double *yj = &s->y[(int64_t)j * G];
            for (int64_t i = 0; i < G; ++i) s->V_mem[i] += lj * yj[i];
        }
    }

    /* 5. V_couple from inbound edges (with cross-dim projection) */
    for (int64_t i = 0; i < G; ++i) s->V_couple[i] = 0.0;
    for (int e = 0; e < n_inbound; ++e) {
        const TriadCouplingEdge *ed = &inbound[e];
        const double *src = rho_snapshots[ed->src_id];
        if (!src) continue;
        int sD = rho_D[ed->src_id];
        int sN = rho_N[ed->src_id];
        const double *src_proj = src;
        double *projected = NULL;
        if (sD != s->D || sN != s->N) {
            projected = triad_mr_project_to_shape(src, sD, sN, s->D, s->N);
            src_proj = projected;
        }
        if (ed->mode == TRIAD_COUPLING_DC_SUBTRACTED) {
            double mean = 0.0;
            for (int64_t i = 0; i < G; ++i) mean += src_proj[i];
            mean /= (double)G;
            double k = ed->kappa;
            for (int64_t i = 0; i < G; ++i)
                s->V_couple[i] += k * (src_proj[i] - mean);
        } else if (ed->mode == TRIAD_COUPLING_PHASE_COHERENT) {
            /* V += κ · Re(Ψ_src · e^{-i k_target · r})
             * Mirrors runtime/multi_runtime.py: reads Ψ_src directly
             * (not ρ), so src must have same grid shape as dst (no
             * cross-dim projection — Python doesn't define one for this
             * mode either). In ND r = x [+ y [+ z]] with isotropic
             * scalar k_target. */
            /* Match runtime/multi_runtime.py: phase_coherent reads
             * src_sub.psi DIRECTLY (the LIVE psi, not the start-of-step
             * snapshot). Since substrates are processed in id order, B's
             * edge from A sees A's psi AFTER A's step — same as Python
             * iterating self.substrates.items() in insertion order. */
            TriadCplx *src_psi = NULL;
            if (ed->src_id >= 0 && ed->src_id < n_all_subs) {
                src_psi = all_subs[ed->src_id]->psi;
            }
            int sD = rho_D[ed->src_id];
            int sN = rho_N[ed->src_id];
            if (src_psi != NULL && sD == s->D && sN == s->N) {
                double k = ed->kappa;
                double kt = ed->k_target;
                if (s->D == 1) {
                    for (int64_t i = 0; i < G; ++i) {
                        double phase = kt * s->x[i];
                        double cphi = cos(phase), sphi = sin(phase);
                        /* Re(Ψ · (cosφ - i sinφ)) = Re*cos + Im*sin */
                        double proj = src_psi[i].re * cphi + src_psi[i].im * sphi;
                        s->V_couple[i] += k * proj;
                    }
                } else if (s->D == 2) {
                    int N = s->N;
                    for (int i = 0; i < N; ++i) {
                        double xi = s->x[i];
                        for (int j = 0; j < N; ++j) {
                            double xj = s->x[j];
                            int64_t idx = (int64_t)i * N + j;
                            double phase = kt * (xi + xj);
                            double cphi = cos(phase), sphi = sin(phase);
                            double proj = src_psi[idx].re * cphi
                                        + src_psi[idx].im * sphi;
                            s->V_couple[idx] += k * proj;
                        }
                    }
                } else { /* 3D */
                    int N = s->N;
                    for (int i = 0; i < N; ++i) {
                        double xi = s->x[i];
                        for (int j = 0; j < N; ++j) {
                            double xj = s->x[j];
                            for (int kk = 0; kk < N; ++kk) {
                                double xk = s->x[kk];
                                int64_t idx = (int64_t)i*N*N + (int64_t)j*N + kk;
                                double phase = kt * (xi + xj + xk);
                                double cphi = cos(phase), sphi = sin(phase);
                                double proj = src_psi[idx].re * cphi
                                            + src_psi[idx].im * sphi;
                                s->V_couple[idx] += k * proj;
                            }
                        }
                    }
                }
            }
        } else {
            double k = ed->kappa;
            for (int64_t i = 0; i < G; ++i)
                s->V_couple[i] += k * src_proj[i];
        }
        free(projected);
    }

    /* 6. Potential phase */
    for (int64_t i = 0; i < G; ++i) {
        double V_tot = s->V_ext_static[i]
                     + s->Lambda_e * rho[i]
                     + s->V_mem[i]
                     + s->V_couple[i];
        double angle = -V_tot * dt_over_hbar;
        TriadCplx phase = { cos(angle), sin(angle) };
        s->psi[i] = _cmul(s->psi[i], phase);
    }

    /* 7. closing OU half-step */
    if (M > 0) {
        for (int64_t i = 0; i < G; ++i) rho[i] = _cabs2(s->psi[i]);
        for (int j = 0; j < M; ++j) {
            double od = s->ou_decay_half[j];
            double od1 = 1.0 - od;
            double *yj = &s->y[(int64_t)j * G];
            for (int64_t i = 0; i < G; ++i) yj[i] = od * yj[i] + od1 * rho[i];
        }
    }

    /* 8. FDT-locked noise */
    if (s->noise_amp > 0.0) {
        double na = s->noise_amp / sqrt(2.0);
        for (int64_t i = 0; i < G; ++i) {
            double xi_r = _randn(rng_state);
            double xi_i = _randn(rng_state);
            s->psi[i].re += na * xi_r;
            s->psi[i].im += na * xi_i;
        }
    }

    /* 9. closing half-linear */
    _ndfft(s, s->psi, s->psi_freq);
    for (int64_t i = 0; i < G; ++i)
        s->psi_freq[i] = _cmul(s->psi_freq[i], s->half_lin[i]);
    _ndifft(s, s->psi_freq, s->psi);

    free(rho);
}

/* ── runtime.run() ND ──────────────────────────────────────────── */

static int _is_active(const TriadSegment *seg, int sid) {
    if (seg->active_ids == NULL) return 1;
    for (int k = 0; k < seg->n_active; ++k) if (seg->active_ids[k] == sid) return 1;
    return 0;
}

static void _ensure_traj_cap(TriadSubstrate *s, int extra_records) {
    int need = s->n_records + extra_records;
    if (need <= s->cap_records) return;
    int new_cap = s->cap_records ? s->cap_records * 2 : 64;
    while (new_cap < need) new_cap *= 2;
    s->density_traj = (double *)realloc(s->density_traj,
        sizeof(double) * (size_t)s->grid_size * (size_t)new_cap);
    s->t_traj = (double *)realloc(s->t_traj, sizeof(double) * (size_t)new_cap);
    s->cap_records = new_cap;
}

void triad_mr_run(TriadMultiRuntime *rt) {
    if (!rt || rt->n_segments == 0) return;

    uint64_t rng_state = 0xDEADBEEFCAFE1234ULL;

    int Ns = rt->n_substrates;
    double **rho_snapshots = (double **)calloc((size_t)Ns, sizeof(double *));

    int *rho_D = (int *)calloc((size_t)Ns, sizeof(int));
    int *rho_N = (int *)calloc((size_t)Ns, sizeof(int));
    for (int i = 0; i < Ns; ++i) {
        int64_t g = rt->substrates[i]->grid_size;
        rho_snapshots[i] = (double *)malloc(sizeof(double) * (size_t)g);
        rho_D[i] = rt->substrates[i]->D;
        rho_N[i] = rt->substrates[i]->N;
    }

    for (int seg_i = 0; seg_i < rt->n_segments; ++seg_i) {
        TriadSegment *seg = &rt->segments[seg_i];

        for (int i = 0; i < Ns; ++i) {
            rt->substrates[i]->active = _is_active(seg, i);
        }

        int n_steps = (int)lrint((seg->t_end - seg->t_start) / rt->dt);
        double seg_t = seg->t_start;

        for (int step = 0; step < n_steps; ++step) {
            /* snapshot ρ AND Ψ of active substrates. The ψ snapshot is
             * needed by phase_coherent coupling, which mirrors
             * runtime/multi_runtime.py reading src_sub.psi directly. */
            for (int i = 0; i < Ns; ++i) {
                if (!rt->substrates[i]->active) continue;
                TriadSubstrate *s = rt->substrates[i];
                for (int64_t x = 0; x < s->grid_size; ++x)
                    rho_snapshots[i][x] = _cabs2(s->psi[x]);
            }

            /* step each active substrate */
            for (int i = 0; i < Ns; ++i) {
                TriadSubstrate *s = rt->substrates[i];
                if (!s->active) continue;

                TriadCouplingEdge *inbound = NULL;
                int n_inbound = 0;
                if (seg->n_edges > 0) {
                    inbound = (TriadCouplingEdge *)malloc(
                        sizeof(TriadCouplingEdge) * (size_t)seg->n_edges);
                    for (int e = 0; e < seg->n_edges; ++e) {
                        if (seg->edges[e].dst_id == i)
                            inbound[n_inbound++] = seg->edges[e];
                    }
                }

                _step_one(s, rho_snapshots,
                          rt->substrates, rt->n_substrates,
                          rho_D, rho_N,
                          inbound, n_inbound, &rng_state);
                free(inbound);

                /* divergence guard (norm = ∫|ψ|² dx^D) */
                double norm = 0.0;
                for (int64_t x = 0; x < s->grid_size; ++x) norm += _cabs2(s->psi[x]);
                norm *= _dx_pow_D(s->dx, s->D);
                double plateau = 0.0;
                if (s->f_FDT_e > 0.0 && s->Gamma_e > 0.0) {
                    double N_grid = (double)s->grid_size;
                    plateau = N_grid * s->f_FDT_e / (2.0 * s->Gamma_e);
                }
                double thresh = 100.0;
                if (10.0 * plateau > thresh) thresh = 10.0 * plateau;
                if (!isfinite(norm) || norm > thresh) {
                    rt->diverged = 1;
                    rt->diverged_segment = seg_i;
                    rt->diverged_step = step;
                    free(rt->diverged_name);
                    rt->diverged_name = _dup_str(s->name);
                    rt->diverged_norm = norm;
                    goto cleanup;
                }
            }

            /* trajectory recording */
            if (step % rt->record_every == 0) {
                for (int i = 0; i < Ns; ++i) {
                    TriadSubstrate *s = rt->substrates[i];
                    _ensure_traj_cap(s, 1);
                    int k = s->n_records;
                    double *dst = &s->density_traj[(int64_t)k * s->grid_size];
                    for (int64_t x = 0; x < s->grid_size; ++x)
                        dst[x] = _cabs2(s->psi[x]);
                    s->t_traj[k] = seg_t;
                    s->n_records++;
                }
            }

            seg_t += rt->dt;
            rt->global_t += rt->dt;
        }
    }

cleanup:
    for (int i = 0; i < Ns; ++i) {
        free(rho_snapshots[i]);
    }
    free(rho_snapshots);
    free(rho_D);
    free(rho_N);
}
