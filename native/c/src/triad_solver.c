#include "triad_rt.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static TriadCplx _cmul(TriadCplx a, TriadCplx b) {
    return (TriadCplx){ a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re };
}

static double _cabs2(TriadCplx a) {
    return a.re*a.re + a.im*a.im;
}

static uint64_t _xs64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static double _randn(uint64_t *state) {
    double u1 = (double)(_xs64(state) >> 11) / (double)(1ULL << 53);
    double u2 = (double)(_xs64(state) >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * TRIAD_PI * u2);
}

static double _maybe_halve_dt(double dt, double Lambda) {
    if (fabs(Lambda) >= 4.0 && dt > 0.0025)
        return 0.0025;
    return dt;
}

typedef struct {
    double Lambda, alpha, Gamma, f_FDT;
    double *lam;
} EffParams;

static EffParams _effective_params(const TriadSolverC *p, double *lam_buf) {
    EffParams e = {0};
    e.Lambda = p->Lambda; e.alpha = p->alpha;
    e.Gamma = p->Gamma; e.f_FDT = p->f_FDT;
    memcpy(lam_buf, p->lam, sizeof(double) * p->M);
    e.lam = lam_buf;
    return e;
}

static void _build_V_ext_1d(const TriadSolverC *p, const double *x, int32_t N,
                            double *V) {
    if (p->V_ext == NULL || strcmp(p->V_ext, "none") == 0) {
        memset(V, 0, sizeof(double) * N);
    } else if (strcmp(p->V_ext, "harmonic") == 0) {
        double coeff = 0.5 * p->m * p->omega * p->omega;
        for (int32_t i = 0; i < N; i++) V[i] = coeff * x[i] * x[i];
    } else if (strcmp(p->V_ext, "double_well") == 0) {
        double w2 = (p->L / 8.0) * (p->L / 8.0);
        for (int32_t i = 0; i < N; i++) {
            double d = x[i] * x[i] - w2;
            V[i] = 0.05 * d * d;
        }
    } else if (strcmp(p->V_ext, "gaussian_bump") == 0) {
        for (int32_t i = 0; i < N; i++)
            V[i] = -2.0 * exp(-x[i] * x[i] / 2.0);
    } else if (strcmp(p->V_ext, "ramp") == 0) {
        for (int32_t i = 0; i < N; i++)
            V[i] = 0.05 * x[i];
    } else if (strcmp(p->V_ext, "lattice") == 0) {
        double k0 = 2.0 * TRIAD_PI / (p->L / 4.0);
        for (int32_t i = 0; i < N; i++)
            V[i] = 0.5 * cos(k0 * x[i]);
    } else {
        memset(V, 0, sizeof(double) * N);
    }
}

static void _build_V_ext_nd(const TriadSolverC *p, const double *xs, int32_t N,
                            int32_t D, double *V) {
    int64_t total = 1;
    for (int32_t d = 0; d < D; d++) total *= N;

    if (p->V_ext == NULL || strcmp(p->V_ext, "none") == 0) {
        memset(V, 0, sizeof(double) * total);
    } else if (strcmp(p->V_ext, "harmonic") == 0) {
        double coeff = 0.5 * p->m * p->omega * p->omega;
        if (D == 2) {
            for (int32_t i = 0; i < N; i++)
                for (int32_t j = 0; j < N; j++)
                    V[(int64_t)i * N + j] = coeff * (xs[i]*xs[i] + xs[j]*xs[j]);
        } else {
            for (int32_t i = 0; i < N; i++)
                for (int32_t j = 0; j < N; j++)
                    for (int32_t k = 0; k < N; k++)
                        V[(int64_t)i*N*N + j*N + k] =
                            coeff * (xs[i]*xs[i] + xs[j]*xs[j] + xs[k]*xs[k]);
        }
    } else if (strcmp(p->V_ext, "double_well") == 0) {
        double w2 = (p->L / 8.0) * (p->L / 8.0);
        if (D == 2) {
            for (int32_t i = 0; i < N; i++)
                for (int32_t j = 0; j < N; j++) {
                    double r2 = xs[i]*xs[i] + xs[j]*xs[j];
                    double d = r2 - w2;
                    V[(int64_t)i * N + j] = 0.05 * d * d;
                }
        } else {
            for (int32_t i = 0; i < N; i++)
                for (int32_t j = 0; j < N; j++)
                    for (int32_t k = 0; k < N; k++) {
                        double r2 = xs[i]*xs[i] + xs[j]*xs[j] + xs[k]*xs[k];
                        double d = r2 - w2;
                        V[(int64_t)i*N*N + j*N + k] = 0.05 * d * d;
                    }
        }
    } else if (strcmp(p->V_ext, "gaussian_bump") == 0) {
        if (D == 2) {
            for (int32_t i = 0; i < N; i++)
                for (int32_t j = 0; j < N; j++) {
                    double r2 = xs[i]*xs[i] + xs[j]*xs[j];
                    V[(int64_t)i * N + j] = -2.0 * exp(-r2 / 2.0);
                }
        } else {
            for (int32_t i = 0; i < N; i++)
                for (int32_t j = 0; j < N; j++)
                    for (int32_t k = 0; k < N; k++) {
                        double r2 = xs[i]*xs[i] + xs[j]*xs[j] + xs[k]*xs[k];
                        V[(int64_t)i*N*N + j*N + k] = -2.0 * exp(-r2 / 2.0);
                    }
        }
    } else if (strcmp(p->V_ext, "ramp") == 0) {
        if (D == 2) {
            for (int32_t i = 0; i < N; i++)
                for (int32_t j = 0; j < N; j++)
                    V[(int64_t)i * N + j] = 0.05 * xs[i];
        } else {
            for (int32_t i = 0; i < N; i++)
                for (int32_t j = 0; j < N; j++)
                    for (int32_t k = 0; k < N; k++)
                        V[(int64_t)i*N*N + j*N + k] = 0.05 * xs[i];
        }
    } else if (strcmp(p->V_ext, "lattice") == 0) {
        double k0 = 2.0 * TRIAD_PI / (p->L / 4.0);
        if (D == 2) {
            for (int32_t i = 0; i < N; i++)
                for (int32_t j = 0; j < N; j++)
                    V[(int64_t)i * N + j] = 0.5 * (cos(k0 * xs[i]) + cos(k0 * xs[j]));
        } else {
            for (int32_t i = 0; i < N; i++)
                for (int32_t j = 0; j < N; j++)
                    for (int32_t k = 0; k < N; k++)
                        V[(int64_t)i*N*N + j*N + k] =
                            0.5 * (cos(k0 * xs[i]) + cos(k0 * xs[j]) + cos(k0 * xs[k]));
        }
    } else {
        memset(V, 0, sizeof(double) * total);
    }
}

static void _build_absorbing_mask_1d(int32_t N, double L, double bc_width_frac,
                                     double *mask) {
    double dx = L / (double)N;
    double border = bc_width_frac * L / 2.0;
    for (int32_t i = 0; i < N; i++) {
        double xi = -L / 2.0 + (double)i * dx;
        if (xi < (-L / 2.0 + border)) {
            double t = (xi - (-L / 2.0 + border)) / (2.0 * border);
            mask[i] = cos(TRIAD_PI * t) * cos(TRIAD_PI * t);
        } else if (xi > (L / 2.0 - border)) {
            double t = (xi - (L / 2.0 - border)) / (2.0 * border);
            mask[i] = cos(TRIAD_PI * t) * cos(TRIAD_PI * t);
        } else {
            mask[i] = 1.0;
        }
    }
}

static bool _is_absorbing_bc(const TriadSolverC *p) {
    return p->bc && strcmp(p->bc, "absorbing") == 0;
}

static TriadSolverResult _solve_1d_impl(const TriadSolverC *p,
                                        const TriadCplx *psi_external,
                                        const double *y_external) {
    int32_t N = p->N;
    double dt = _maybe_halve_dt(p->dt, p->Lambda);
    double dx = p->L / (double)N;

    double *x = malloc(sizeof(double) * N);
    for (int32_t i = 0; i < N; i++)
        x[i] = -p->L / 2.0 + (double)i * dx;

    double *k = malloc(sizeof(double) * N);
    triad_fftfreq(N, dx, k);

    double *lam_buf = malloc(sizeof(double) * p->M);
    EffParams eff = _effective_params(p, lam_buf);

    double *V_ext = malloc(sizeof(double) * N);
    _build_V_ext_1d(p, x, N, V_ext);

    TriadCplx *psi = malloc(sizeof(TriadCplx) * N);
    if (psi_external) {
        memcpy(psi, psi_external, sizeof(TriadCplx) * N);
    } else if (p->init_mode == 1) {

        uint64_t init_rng = (p->seed ? p->seed : 0xDEADBEEFCAFE1234ULL) ^ 0x9E3779B97F4A7C15ULL;
        double norm = 0;
        for (int32_t i = 0; i < N; i++) {
            psi[i] = (TriadCplx){ _randn(&init_rng), _randn(&init_rng) };
            norm += _cabs2(psi[i]);
        }
        norm = sqrt(norm * dx);
        for (int32_t i = 0; i < N; i++) {
            psi[i].re /= norm;
            psi[i].im /= norm;
        }
    } else {
        double s = p->init_sigma > 0 ? p->init_sigma : 2.0;
        double k0x = p->init_k0[0];
        double norm = 0;
        for (int32_t i = 0; i < N; i++) {
            double g = exp(-x[i] * x[i] / (2.0 * s * s));
            double ph = k0x * x[i];
            psi[i] = (TriadCplx){ g * cos(ph), g * sin(ph) };
            norm += g * g;
        }
        norm = sqrt(norm * dx);
        for (int32_t i = 0; i < N; i++) {
            psi[i].re /= norm;
            psi[i].im /= norm;
        }
    }

    int32_t M = p->M;
    double *y = NULL;
    double *ou_decay_half = NULL;
    if (M > 0) {
        y = calloc(M * N, sizeof(double));
        if (y_external) memcpy(y, y_external, (size_t)M * N * sizeof(double));
        ou_decay_half = malloc(sizeof(double) * M);
        for (int32_t j = 0; j < M; j++)
            ou_decay_half[j] = exp(-p->nu[j] * dt * 0.5);
    }

    TriadCplx *half_lin = malloc(sizeof(TriadCplx) * N);
    for (int32_t i = 0; i < N; i++) {
        double abs_k = fabs(k[i]);
        double H_lin = p->hbar * p->hbar * k[i] * k[i] / (2.0 * p->m)
                     + eff.alpha * pow(abs_k > 0 ? abs_k : 1e-30, p->sigma);
        double angle = (-H_lin / p->hbar) * (dt / 2.0);
        double decay = (-eff.Gamma / p->hbar) * (dt / 2.0);
        double mag = exp(decay);
        half_lin[i] = (TriadCplx){ mag * cos(angle), mag * sin(angle) };
    }

    double noise_amp = 0.0;
    uint64_t rng_state = p->seed ? p->seed : 0xDEADBEEFCAFE1234ULL;
    {

        double f_eff = eff.f_FDT;
        if (p->fdt_couple && eff.Gamma > 0)
            f_eff = 2.0 * eff.Gamma * dx * p->kT / p->hbar;
        if (f_eff < 1e-12)
            f_eff = 1e-12;
        noise_amp = sqrt(f_eff * dt / dx);
    }

    TriadCplx *psi_k = malloc(sizeof(TriadCplx) * N);
    double *rho = malloc(sizeof(double) * N);

    double *bc_mask = NULL;
    if (_is_absorbing_bc(p)) {
        bc_mask = malloc(sizeof(double) * N);
        _build_absorbing_mask_1d(N, p->L, p->bc_width > 0 ? p->bc_width : 0.15, bc_mask);
    }

    int32_t n_steps = (int32_t)round(p->T / dt);
    double dt_over_hbar = dt / p->hbar;

    int32_t rec_every = p->record_every > 0 ? p->record_every : 4;
    int32_t max_recs = n_steps / rec_every + 2;
    double *density_t = malloc(sizeof(double) * (size_t)N * (size_t)max_recs);
    int32_t n_rec = 0;

    for (int32_t step = 0; step <= n_steps; step++) {
        for (int32_t i = 0; i < N; i++)
            rho[i] = _cabs2(psi[i]);
        if (step % rec_every == 0 || step == n_steps) {
            memcpy(density_t + (size_t)n_rec * (size_t)N, rho,
                   sizeof(double) * (size_t)N);
            n_rec++;
        }
        if (step == n_steps) break;

        triad_fft_fn(N, psi, psi_k);
        for (int32_t i = 0; i < N; i++)
            psi_k[i] = _cmul(psi_k[i], half_lin[i]);
        triad_ifft_fn(N, psi_k, psi);

        for (int32_t i = 0; i < N; i++)
            rho[i] = _cabs2(psi[i]);

        if (M > 0) {
            for (int32_t j = 0; j < M; j++) {
                double od = ou_decay_half[j];
                double od1 = 1.0 - od;
                for (int32_t i = 0; i < N; i++)
                    y[j * N + i] = od * y[j * N + i] + od1 * rho[i];
            }
        }

        for (int32_t i = 0; i < N; i++) {
            double V_mem = 0.0;
            if (M > 0)
                for (int32_t j = 0; j < M; j++)
                    V_mem += eff.lam[j] * y[j * N + i];
            double V_total = V_ext[i] + eff.Lambda * rho[i] + V_mem;
            double angle = -V_total * dt_over_hbar;
            TriadCplx phase = { cos(angle), sin(angle) };
            psi[i] = _cmul(psi[i], phase);
        }

        if (M > 0) {
            for (int32_t i = 0; i < N; i++)
                rho[i] = _cabs2(psi[i]);
            for (int32_t j = 0; j < M; j++) {
                double od = ou_decay_half[j];
                double od1 = 1.0 - od;
                for (int32_t i = 0; i < N; i++)
                    y[j * N + i] = od * y[j * N + i] + od1 * rho[i];
            }
        }

        {
            double na = noise_amp / sqrt(2.0);
            for (int32_t i = 0; i < N; i++) {
                double xi_r = _randn(&rng_state);
                double xi_i = _randn(&rng_state);
                psi[i].re += na * xi_r;
                psi[i].im += na * xi_i;
            }
        }

        triad_fft_fn(N, psi, psi_k);
        for (int32_t i = 0; i < N; i++)
            psi_k[i] = _cmul(psi_k[i], half_lin[i]);
        triad_ifft_fn(N, psi_k, psi);

        if (bc_mask) {
            for (int32_t i = 0; i < N; i++) {
                psi[i].re *= bc_mask[i];
                psi[i].im *= bc_mask[i];
            }
        }
    }

    TriadSolverResult result;
    result.psi_final = psi;
    result.y_final = y;
    result.density_final = rho;
    result.x = x;
    result.dx = dx;
    result.density_t = density_t;
    result.n_records = n_rec;

    free(psi_k);
    free(k);
    free(V_ext);
    free(half_lin);
    free(lam_buf);
    free(ou_decay_half);
    free(bc_mask);

    return result;
}

void triad_solver_result_free(TriadSolverResult *r) {
    if (!r) return;
    free(r->psi_final);
    free(r->y_final);
    free(r->density_final);
    free(r->x);
    free(r->density_t);
}

static bool _validate_triad_input(const TriadSolverC *p) {
    if (!p) return false;
    if (p->M < 3) return false;
    if (p->mode != 2) return false;
    if (p->Gamma <= 0.0) return false;
    if (p->f_FDT <= 0.0) return false;
    if (p->kT <= 0.0) return false;
    if (p->N < 8) return false;
    if (!(p->L > 0.0)) return false;
    if (!(p->dt > 0.0)) return false;
    if (!(p->T > 0.0)) return false;
    if (!(p->hbar > 0.0)) return false;
    if (!(p->m > 0.0)) return false;
    if (!p->nu || !p->lam) return false;
    return true;
}

TriadSolverResult triad_solve_1d(const TriadSolverC *p) {
    if (!_validate_triad_input(p)) {
        TriadSolverResult r = {0};
        return r;
    }
    return _solve_1d_impl(p, NULL, NULL);
}

TriadSolverResult triad_solve_from_psi(const TriadSolverC *p, const TriadCplx *psi_init) {
    if (!_validate_triad_input(p)) {
        TriadSolverResult r = {0};
        return r;
    }
    return _solve_1d_impl(p, psi_init, NULL);
}

TriadSolverResult triad_solve_from_state(const TriadSolverC *p, const TriadCplx *psi_init,
                                         const double *y_init) {
    if (!_validate_triad_input(p)) {
        TriadSolverResult r = {0};
        return r;
    }
    return _solve_1d_impl(p, psi_init, y_init);
}

TriadSolverResult2D triad_solve_2d(const TriadSolverC *p) {
    if (!_validate_triad_input(p)) {
        TriadSolverResult2D r = {0};
        return r;
    }
    int32_t N = p->N;
    double dt = _maybe_halve_dt(p->dt, p->Lambda);
    double dx = p->L / (double)N;
    int64_t N2 = (int64_t)N * N;

    double *xs = malloc(sizeof(double) * N);
    for (int32_t i = 0; i < N; i++)
        xs[i] = -p->L / 2.0 + (double)i * dx;

    double *kvec = malloc(sizeof(double) * N);
    triad_fftfreq(N, dx, kvec);

    double *k2 = malloc(sizeof(double) * N2);
    double *abs_k = malloc(sizeof(double) * N2);
    for (int32_t i = 0; i < N; i++)
        for (int32_t j = 0; j < N; j++) {
            int64_t idx = (int64_t)i * N + j;
            k2[idx] = kvec[i] * kvec[i] + kvec[j] * kvec[j];
            abs_k[idx] = sqrt(k2[idx]);
        }

    double *lam_buf = malloc(sizeof(double) * p->M);
    EffParams eff = _effective_params(p, lam_buf);

    double *V_ext = malloc(sizeof(double) * N2);
    _build_V_ext_nd(p, xs, N, 2, V_ext);

    TriadCplx *psi = malloc(sizeof(TriadCplx) * N2);
    if (p->init_mode == 1) {
        uint64_t init_rng = (p->seed ? p->seed : 0xDEADBEEFCAFE1234ULL) ^ 0x9E3779B97F4A7C15ULL;
        double norm = 0;
        for (int64_t i = 0; i < N2; i++) {
            psi[i] = (TriadCplx){ _randn(&init_rng), _randn(&init_rng) };
            norm += _cabs2(psi[i]);
        }
        norm = sqrt(norm * dx * dx);
        for (int64_t i = 0; i < N2; i++) {
            psi[i].re /= norm;
            psi[i].im /= norm;
        }
    } else {
        double s = p->init_sigma > 0 ? p->init_sigma : 2.0;
        double k0x = p->init_k0[0], k0y = p->init_k0[1];
        double norm = 0;
        for (int32_t i = 0; i < N; i++)
            for (int32_t j = 0; j < N; j++) {
                int64_t idx = (int64_t)i * N + j;
                double g = exp(-(xs[i]*xs[i] + xs[j]*xs[j]) / (2.0 * s * s));
                double ph = k0x * xs[i] + k0y * xs[j];
                psi[idx] = (TriadCplx){ g * cos(ph), g * sin(ph) };
                norm += g * g;
            }
        norm = sqrt(norm * dx * dx);
        for (int64_t i = 0; i < N2; i++) {
            psi[i].re /= norm;
            psi[i].im /= norm;
        }
    }

    int32_t M = p->M;
    double *y = NULL;
    double *ou_decay_half = NULL;
    if (M > 0) {
        y = calloc(M * N2, sizeof(double));
        ou_decay_half = malloc(sizeof(double) * M);
        for (int32_t j = 0; j < M; j++)
            ou_decay_half[j] = exp(-p->nu[j] * dt * 0.5);
    }

    TriadCplx *half_lin = malloc(sizeof(TriadCplx) * N2);
    for (int64_t i = 0; i < N2; i++) {
        double ak = abs_k[i] > 0 ? abs_k[i] : 1e-30;
        double H_lin = p->hbar * p->hbar * k2[i] / (2.0 * p->m)
                     + eff.alpha * pow(ak, p->sigma);
        double angle = (-H_lin / p->hbar) * (dt / 2.0);
        double decay = (-eff.Gamma / p->hbar) * (dt / 2.0);
        double mag = exp(decay);
        half_lin[i] = (TriadCplx){ mag * cos(angle), mag * sin(angle) };
    }

    double noise_amp = 0.0;
    uint64_t rng_state = p->seed ? p->seed : 0xDEADBEEFCAFE1234ULL;
    {
        double f_eff = eff.f_FDT;
        if (p->fdt_couple && eff.Gamma > 0)
            f_eff = 2.0 * eff.Gamma * (dx * dx) * p->kT / p->hbar;
        if (f_eff < 1e-12)
            f_eff = 1e-12;
        noise_amp = sqrt(f_eff * dt / (dx * dx));
    }

    TriadCplx *psi_k = malloc(sizeof(TriadCplx) * N2);
    double *rho = malloc(sizeof(double) * N2);

    double *bc_mask = NULL;
    if (_is_absorbing_bc(p)) {
        double *mask1d = malloc(sizeof(double) * N);
        _build_absorbing_mask_1d(N, p->L, p->bc_width > 0 ? p->bc_width : 0.15, mask1d);
        bc_mask = malloc(sizeof(double) * N2);
        for (int32_t i = 0; i < N; i++)
            for (int32_t j = 0; j < N; j++)
                bc_mask[(int64_t)i * N + j] = mask1d[i] * mask1d[j];
        free(mask1d);
    }

    int32_t n_steps = (int32_t)round(p->T / dt);
    double dt_over_hbar = dt / p->hbar;

    for (int32_t step = 0; step < n_steps; step++) {
        triad_fft2_fn(N, psi, psi_k);
        for (int64_t i = 0; i < N2; i++)
            psi_k[i] = _cmul(psi_k[i], half_lin[i]);
        triad_ifft2_fn(N, psi_k, psi);

        triad_density_64(N2, psi, rho);

        if (M > 0) {
            for (int32_t j = 0; j < M; j++) {
                double od = ou_decay_half[j];
                double od1 = 1.0 - od;
                for (int64_t i = 0; i < N2; i++)
                    y[j * N2 + i] = od * y[j * N2 + i] + od1 * rho[i];
            }
        }

        for (int64_t i = 0; i < N2; i++) {
            double V_mem = 0.0;
            if (M > 0)
                for (int32_t j = 0; j < M; j++)
                    V_mem += eff.lam[j] * y[j * N2 + i];
            double V_total = V_ext[i] + eff.Lambda * rho[i] + V_mem;
            double angle = -V_total * dt_over_hbar;
            TriadCplx phase = { cos(angle), sin(angle) };
            psi[i] = _cmul(psi[i], phase);
        }

        triad_density_64(N2, psi, rho);

        if (M > 0) {
            for (int32_t j = 0; j < M; j++) {
                double od = ou_decay_half[j];
                double od1 = 1.0 - od;
                for (int64_t i = 0; i < N2; i++)
                    y[j * N2 + i] = od * y[j * N2 + i] + od1 * rho[i];
            }
        }

        {
            double na = noise_amp / sqrt(2.0);
            for (int64_t i = 0; i < N2; i++) {
                double xi_r = _randn(&rng_state);
                double xi_i = _randn(&rng_state);
                psi[i].re += na * xi_r;
                psi[i].im += na * xi_i;
            }
        }

        triad_fft2_fn(N, psi, psi_k);
        for (int64_t i = 0; i < N2; i++)
            psi_k[i] = _cmul(psi_k[i], half_lin[i]);
        triad_ifft2_fn(N, psi_k, psi);

        if (bc_mask) {
            for (int64_t i = 0; i < N2; i++) {
                psi[i].re *= bc_mask[i];
                psi[i].im *= bc_mask[i];
            }
        }
    }

    triad_density_64(N2, psi, rho);

    TriadSolverResult2D result;
    result.psi_final = psi;
    result.y_final = y;
    result.density_final = rho;
    result.x = xs;
    result.dx = dx;

    free(psi_k);
    free(kvec);
    free(k2);
    free(abs_k);
    free(V_ext);
    free(half_lin);
    free(lam_buf);
    free(ou_decay_half);
    free(bc_mask);

    return result;
}

void triad_solver_result_2d_free(TriadSolverResult2D *r) {
    if (!r) return;
    free(r->psi_final);
    free(r->y_final);
    free(r->density_final);
    free(r->x);
}

TriadSolverResult3D triad_solve_3d(const TriadSolverC *p) {
    if (!_validate_triad_input(p)) {
        TriadSolverResult3D r = {0};
        return r;
    }
    int32_t N = p->N;
    double dt = _maybe_halve_dt(p->dt, p->Lambda);
    double dx = p->L / (double)N;
    int64_t N3 = (int64_t)N * N * N;

    double *xs = malloc(sizeof(double) * N);
    for (int32_t i = 0; i < N; i++)
        xs[i] = -p->L / 2.0 + (double)i * dx;

    double *kvec = malloc(sizeof(double) * N);
    triad_fftfreq(N, dx, kvec);

    double *k2 = malloc(sizeof(double) * N3);
    double *abs_k = malloc(sizeof(double) * N3);
    for (int32_t i = 0; i < N; i++)
        for (int32_t j = 0; j < N; j++)
            for (int32_t k = 0; k < N; k++) {
                int64_t idx = (int64_t)i*N*N + j*N + k;
                k2[idx] = kvec[i]*kvec[i] + kvec[j]*kvec[j] + kvec[k]*kvec[k];
                abs_k[idx] = sqrt(k2[idx]);
            }

    double *lam_buf = malloc(sizeof(double) * p->M);
    EffParams eff = _effective_params(p, lam_buf);

    double *V_ext = malloc(sizeof(double) * N3);
    _build_V_ext_nd(p, xs, N, 3, V_ext);

    TriadCplx *psi = malloc(sizeof(TriadCplx) * N3);
    if (p->init_mode == 1) {
        uint64_t init_rng = (p->seed ? p->seed : 0xDEADBEEFCAFE1234ULL) ^ 0x9E3779B97F4A7C15ULL;
        double norm = 0;
        for (int64_t i = 0; i < N3; i++) {
            psi[i] = (TriadCplx){ _randn(&init_rng), _randn(&init_rng) };
            norm += _cabs2(psi[i]);
        }
        norm = sqrt(norm * dx * dx * dx);
        for (int64_t i = 0; i < N3; i++) {
            psi[i].re /= norm;
            psi[i].im /= norm;
        }
    } else {
        double s = p->init_sigma > 0 ? p->init_sigma : 2.0;
        double k0x = p->init_k0[0], k0y = p->init_k0[1], k0z = p->init_k0[2];
        double norm = 0;
        for (int32_t i = 0; i < N; i++)
            for (int32_t j = 0; j < N; j++)
                for (int32_t k = 0; k < N; k++) {
                    int64_t idx = (int64_t)i*N*N + j*N + k;
                    double g = exp(-(xs[i]*xs[i] + xs[j]*xs[j] + xs[k]*xs[k]) / (2.0 * s * s));
                    double ph = k0x * xs[i] + k0y * xs[j] + k0z * xs[k];
                    psi[idx] = (TriadCplx){ g * cos(ph), g * sin(ph) };
                    norm += g * g;
                }
        norm = sqrt(norm * dx * dx * dx);
        for (int64_t i = 0; i < N3; i++) {
            psi[i].re /= norm;
            psi[i].im /= norm;
        }
    }

    int32_t M = p->M;
    double *y = NULL;
    double *ou_decay_half = NULL;
    if (M > 0) {
        y = calloc(M * N3, sizeof(double));
        ou_decay_half = malloc(sizeof(double) * M);
        for (int32_t j = 0; j < M; j++)
            ou_decay_half[j] = exp(-p->nu[j] * dt * 0.5);
    }

    TriadCplx *half_lin = malloc(sizeof(TriadCplx) * N3);
    for (int64_t i = 0; i < N3; i++) {
        double ak = abs_k[i] > 0 ? abs_k[i] : 1e-30;
        double H_lin = p->hbar * p->hbar * k2[i] / (2.0 * p->m)
                     + eff.alpha * pow(ak, p->sigma);
        double angle = (-H_lin / p->hbar) * (dt / 2.0);
        double decay = (-eff.Gamma / p->hbar) * (dt / 2.0);
        double mag = exp(decay);
        half_lin[i] = (TriadCplx){ mag * cos(angle), mag * sin(angle) };
    }

    double noise_amp = 0.0;
    uint64_t rng_state = p->seed ? p->seed : 0xDEADBEEFCAFE1234ULL;
    {
        double f_eff = eff.f_FDT;
        if (p->fdt_couple && eff.Gamma > 0)
            f_eff = 2.0 * eff.Gamma * (dx * dx * dx) * p->kT / p->hbar;
        if (f_eff < 1e-12)
            f_eff = 1e-12;
        noise_amp = sqrt(f_eff * dt / (dx * dx * dx));
    }

    TriadCplx *psi_k = malloc(sizeof(TriadCplx) * N3);
    double *rho = malloc(sizeof(double) * N3);

    double *bc_mask = NULL;
    if (_is_absorbing_bc(p)) {
        double *mask1d = malloc(sizeof(double) * N);
        _build_absorbing_mask_1d(N, p->L, p->bc_width > 0 ? p->bc_width : 0.15, mask1d);
        bc_mask = malloc(sizeof(double) * N3);
        for (int32_t i = 0; i < N; i++)
            for (int32_t j = 0; j < N; j++)
                for (int32_t k = 0; k < N; k++)
                    bc_mask[(int64_t)i*N*N + j*N + k] = mask1d[i] * mask1d[j] * mask1d[k];
        free(mask1d);
    }

    int32_t n_steps = (int32_t)round(p->T / dt);
    double dt_over_hbar = dt / p->hbar;

    int32_t rec_every = p->record_every > 0 ? p->record_every : 4;
    int32_t max_recs = n_steps / rec_every + 2;
    double *peak_t = malloc(sizeof(double) * max_recs);
    double *part_t = malloc(sizeof(double) * max_recs);
    int32_t n_rec = 0;

    for (int32_t step = 0; step <= n_steps; step++) {
        triad_density_64(N3, psi, rho);
        if (step % rec_every == 0 || step == n_steps) {
            double peak = 0;
            double norm2 = 0;
            double rho2_sum = 0;
            for (int64_t i = 0; i < N3; i++) {
                if (rho[i] > peak) peak = rho[i];
                norm2 += rho[i];
                rho2_sum += rho[i] * rho[i];
            }
            norm2 *= dx * dx * dx;
            rho2_sum *= dx * dx * dx;
            double pr = (rho2_sum > 1e-300) ? norm2 * norm2 / rho2_sum : 0;
            peak_t[n_rec] = peak;
            part_t[n_rec] = pr;
            n_rec++;
        }
        if (step == n_steps) break;

        triad_fft3_fn(N, psi, psi_k);
        for (int64_t i = 0; i < N3; i++)
            psi_k[i] = _cmul(psi_k[i], half_lin[i]);
        triad_ifft3_fn(N, psi_k, psi);

        triad_density_64(N3, psi, rho);

        if (M > 0) {
            for (int32_t j = 0; j < M; j++) {
                double od = ou_decay_half[j];
                double od1 = 1.0 - od;
                for (int64_t i = 0; i < N3; i++)
                    y[j * N3 + i] = od * y[j * N3 + i] + od1 * rho[i];
            }
        }

        for (int64_t i = 0; i < N3; i++) {
            double V_mem = 0.0;
            if (M > 0)
                for (int32_t j = 0; j < M; j++)
                    V_mem += eff.lam[j] * y[j * N3 + i];
            double V_total = V_ext[i] + eff.Lambda * rho[i] + V_mem;
            double angle = -V_total * dt_over_hbar;
            TriadCplx phase = { cos(angle), sin(angle) };
            psi[i] = _cmul(psi[i], phase);
        }

        triad_density_64(N3, psi, rho);

        if (M > 0) {
            for (int32_t j = 0; j < M; j++) {
                double od = ou_decay_half[j];
                double od1 = 1.0 - od;
                for (int64_t i = 0; i < N3; i++)
                    y[j * N3 + i] = od * y[j * N3 + i] + od1 * rho[i];
            }
        }

        {
            double na = noise_amp / sqrt(2.0);
            for (int64_t i = 0; i < N3; i++) {
                double xi_r = _randn(&rng_state);
                double xi_i = _randn(&rng_state);
                psi[i].re += na * xi_r;
                psi[i].im += na * xi_i;
            }
        }

        triad_fft3_fn(N, psi, psi_k);
        for (int64_t i = 0; i < N3; i++)
            psi_k[i] = _cmul(psi_k[i], half_lin[i]);
        triad_ifft3_fn(N, psi_k, psi);

        if (bc_mask) {
            for (int64_t i = 0; i < N3; i++) {
                psi[i].re *= bc_mask[i];
                psi[i].im *= bc_mask[i];
            }
        }
    }

    triad_density_64(N3, psi, rho);

    TriadSolverResult3D result;
    result.psi_final = psi;
    result.y_final = y;
    result.density_final = rho;
    result.x = xs;
    result.dx = dx;
    result.peak_t = peak_t;
    result.participation_t = part_t;
    result.n_records = n_rec;

    free(psi_k);
    free(kvec);
    free(k2);
    free(abs_k);
    free(V_ext);
    free(half_lin);
    free(lam_buf);
    free(ou_decay_half);
    free(bc_mask);

    return result;
}

void triad_solver_result_3d_free(TriadSolverResult3D *r) {
    if (!r) return;
    free(r->psi_final);
    free(r->y_final);
    free(r->density_final);
    free(r->x);
    free(r->peak_t);
    free(r->participation_t);
}

static int64_t _pow_i64(int32_t base, int32_t exp) {
    int64_t out = 1;
    for (int32_t i = 0; i < exp; i++)
        out *= (int64_t)base;
    return out;
}

static void _build_strides(int32_t N, int32_t D, int64_t *strides) {
    int64_t stride = 1;
    for (int32_t d = D - 1; d >= 0; d--) {
        strides[d] = stride;
        stride *= (int64_t)N;
    }
}

static int32_t _coord_at(int64_t idx, int32_t axis, int32_t N,
                         const int64_t *strides) {
    return (int32_t)((idx / strides[axis]) % (int64_t)N);
}

static void _build_V_ext_nd_generic(const TriadSolverC *p, const double *xs,
                                    const int64_t *strides, int64_t total,
                                    double *V) {
    int32_t N = p->N;
    int32_t D = p->D;
    if (p->V_ext == NULL || strcmp(p->V_ext, "none") == 0) {
        memset(V, 0, sizeof(double) * total);
        return;
    }

    double coeff = 0.5 * p->m * p->omega * p->omega;
    double w2 = (p->L / 8.0) * (p->L / 8.0);
    double k0 = 2.0 * TRIAD_PI / (p->L / 4.0);
    for (int64_t idx = 0; idx < total; idx++) {
        double r2 = 0.0;
        double lattice = 0.0;
        int32_t c0 = _coord_at(idx, 0, N, strides);
        for (int32_t d = 0; d < D; d++) {
            double x = xs[_coord_at(idx, d, N, strides)];
            r2 += x * x;
            lattice += cos(k0 * x);
        }

        if (strcmp(p->V_ext, "harmonic") == 0) {
            V[idx] = coeff * r2;
        } else if (strcmp(p->V_ext, "double_well") == 0) {
            double delta = r2 - w2;
            V[idx] = 0.05 * delta * delta;
        } else if (strcmp(p->V_ext, "gaussian_bump") == 0) {
            V[idx] = -2.0 * exp(-r2 / 2.0);
        } else if (strcmp(p->V_ext, "ramp") == 0) {
            V[idx] = 0.05 * xs[c0];
        } else if (strcmp(p->V_ext, "lattice") == 0) {
            V[idx] = 0.5 * lattice;
        } else {
            V[idx] = 0.0;
        }
    }
}

static void _fft_nd_axis(int32_t N, int32_t D, int32_t axis,
                         const int64_t *strides, int64_t total,
                         const TriadCplx *in, TriadCplx *out,
                         TriadCplx *line_in, TriadCplx *line_out,
                         int inverse) {
    int64_t stride = strides[axis];
    for (int64_t base = 0; base < total; base++) {
        if (((base / stride) % (int64_t)N) != 0)
            continue;
        for (int32_t i = 0; i < N; i++)
            line_in[i] = in[base + (int64_t)i * stride];
        if (inverse)
            triad_ifft_fn(N, line_in, line_out);
        else
            triad_fft_fn(N, line_in, line_out);
        for (int32_t i = 0; i < N; i++)
            out[base + (int64_t)i * stride] = line_out[i];
    }
    (void)D;
}

static void _fft_nd_apply(int32_t N, int32_t D, const int64_t *strides,
                          int64_t total, const TriadCplx *in,
                          TriadCplx *out, TriadCplx *work,
                          TriadCplx *line_in, TriadCplx *line_out,
                          int inverse) {
    const TriadCplx *src = in;
    TriadCplx *dst = out;
    for (int32_t axis = 0; axis < D; axis++) {
        _fft_nd_axis(N, D, axis, strides, total, src, dst,
                     line_in, line_out, inverse);
        src = dst;
        dst = (dst == out) ? work : out;
    }
    if (src != out)
        memcpy(out, src, sizeof(TriadCplx) * total);
}

TriadSolverResultND triad_solve_nd(const TriadSolverC *p) {
    TriadSolverResultND empty;
    memset(&empty, 0, sizeof(empty));
    if (!p || p->D < 4)
        return empty;

    int32_t N = p->N;
    int32_t D = p->D;
    double dt = _maybe_halve_dt(p->dt, p->Lambda);
    double dx = p->L / (double)N;
    int64_t total = _pow_i64(N, D);
    double dV = pow(dx, (double)D);

    int64_t *strides = malloc(sizeof(int64_t) * D);
    _build_strides(N, D, strides);

    double *xs = malloc(sizeof(double) * N);
    for (int32_t i = 0; i < N; i++)
        xs[i] = -p->L / 2.0 + (double)i * dx;

    double *kvec = malloc(sizeof(double) * N);
    triad_fftfreq(N, dx, kvec);

    double *lam_buf = malloc(sizeof(double) * p->M);
    EffParams eff = _effective_params(p, lam_buf);

    double *V_ext = malloc(sizeof(double) * total);
    _build_V_ext_nd_generic(p, xs, strides, total, V_ext);

    double *k2 = malloc(sizeof(double) * total);
    double *abs_k = malloc(sizeof(double) * total);
    double *r2_buf = malloc(sizeof(double) * total);
    for (int64_t idx = 0; idx < total; idx++) {
        double kk = 0.0;
        double rr = 0.0;
        for (int32_t d = 0; d < D; d++) {
            int32_t c = _coord_at(idx, d, N, strides);
            kk += kvec[c] * kvec[c];
            rr += xs[c] * xs[c];
        }
        k2[idx] = kk;
        abs_k[idx] = sqrt(kk);
        r2_buf[idx] = rr;
    }

    TriadCplx *psi = malloc(sizeof(TriadCplx) * total);
    if (p->init_mode == 1) {
        uint64_t init_rng = (p->seed ? p->seed : 0xDEADBEEFCAFE1234ULL) ^ 0x9E3779B97F4A7C15ULL;
        double norm = 0.0;
        for (int64_t i = 0; i < total; i++) {
            psi[i] = (TriadCplx){ _randn(&init_rng), _randn(&init_rng) };
            norm += _cabs2(psi[i]);
        }
        norm = sqrt(norm * dV);
        for (int64_t i = 0; i < total; i++) {
            psi[i].re /= norm;
            psi[i].im /= norm;
        }
    } else {
        double s = p->init_sigma > 0 ? p->init_sigma : 2.0;
        double norm = 0.0;
        for (int64_t idx = 0; idx < total; idx++) {
            double phase = 0.0;
            for (int32_t d = 0; d < D; d++) {
                double k0d;
                if (p->init_k0_ext && p->init_k0_ext_len > 0)
                    k0d = d < p->init_k0_ext_len ? p->init_k0_ext[d] : 0.0;
                else
                    k0d = d < 3 ? p->init_k0[d] : 0.0;
                if (k0d == 0.0) continue;
                int32_t c = _coord_at(idx, d, N, strides);
                phase += k0d * xs[c];
            }
            double g = exp(-r2_buf[idx] / (2.0 * s * s));
            psi[idx] = (TriadCplx){ g * cos(phase), g * sin(phase) };
            norm += g * g;
        }
        norm = sqrt(norm * dV);
        for (int64_t i = 0; i < total; i++) {
            psi[i].re /= norm;
            psi[i].im /= norm;
        }
    }

    int32_t M = p->M;
    double *y = NULL;
    double *ou_decay_half = NULL;
    if (M > 0) {
        y = calloc((size_t)M * (size_t)total, sizeof(double));
        ou_decay_half = malloc(sizeof(double) * M);
        for (int32_t j = 0; j < M; j++)
            ou_decay_half[j] = exp(-p->nu[j] * dt * 0.5);
    }

    TriadCplx *half_lin = malloc(sizeof(TriadCplx) * total);
    for (int64_t i = 0; i < total; i++) {
        double ak = abs_k[i] > 0 ? abs_k[i] : 1e-30;
        double H_lin = p->hbar * p->hbar * k2[i] / (2.0 * p->m)
                     + eff.alpha * pow(ak, p->sigma);
        double angle = (-H_lin / p->hbar) * (dt / 2.0);
        double decay = (-eff.Gamma / p->hbar) * (dt / 2.0);
        double mag = exp(decay);
        half_lin[i] = (TriadCplx){ mag * cos(angle), mag * sin(angle) };
    }

    double noise_amp = 0.0;
    uint64_t rng_state = p->seed ? p->seed : 0xDEADBEEFCAFE1234ULL;
    {
        double f_eff = eff.f_FDT;
        if (p->fdt_couple && eff.Gamma > 0)
            f_eff = 2.0 * eff.Gamma * dV * p->kT / p->hbar;
        if (f_eff < 1e-12)
            f_eff = 1e-12;
        noise_amp = sqrt(f_eff * dt / dV);
    }

    TriadCplx *psi_k = malloc(sizeof(TriadCplx) * total);
    TriadCplx *psi_work = malloc(sizeof(TriadCplx) * total);
    TriadCplx *line_in = malloc(sizeof(TriadCplx) * N);
    TriadCplx *line_out = malloc(sizeof(TriadCplx) * N);
    double *rho = malloc(sizeof(double) * total);

    double *bc_mask = NULL;
    if (_is_absorbing_bc(p)) {
        double *mask1d = malloc(sizeof(double) * N);
        _build_absorbing_mask_1d(N, p->L, p->bc_width > 0 ? p->bc_width : 0.15, mask1d);
        bc_mask = malloc(sizeof(double) * total);
        for (int64_t idx = 0; idx < total; idx++) {
            double m = 1.0;
            for (int32_t d = 0; d < D; d++)
                m *= mask1d[_coord_at(idx, d, N, strides)];
            bc_mask[idx] = m;
        }
        free(mask1d);
    }

    int32_t n_steps = (int32_t)round(p->T / dt);
    double dt_over_hbar = dt / p->hbar;
    int32_t rec_every = p->record_every > 0 ? p->record_every : 4;
    int32_t max_recs = n_steps / rec_every + 2;
    double *peak_t = malloc(sizeof(double) * max_recs);
    double *part_t = malloc(sizeof(double) * max_recs);
    int32_t n_rec = 0;

    for (int32_t step = 0; step <= n_steps; step++) {
        triad_density_64(total, psi, rho);
        if (step % rec_every == 0 || step == n_steps) {
            double peak = 0.0;
            double norm2 = 0.0;
            double rho2_sum = 0.0;
            for (int64_t i = 0; i < total; i++) {
                if (rho[i] > peak) peak = rho[i];
                norm2 += rho[i];
                rho2_sum += rho[i] * rho[i];
            }
            norm2 *= dV;
            rho2_sum *= dV;
            peak_t[n_rec] = peak;
            part_t[n_rec] = (rho2_sum > 1e-300) ? norm2 * norm2 / rho2_sum : 0.0;
            n_rec++;
        }
        if (step == n_steps)
            break;

        _fft_nd_apply(N, D, strides, total, psi, psi_k, psi_work,
                      line_in, line_out, 0);
        for (int64_t i = 0; i < total; i++)
            psi_k[i] = _cmul(psi_k[i], half_lin[i]);
        _fft_nd_apply(N, D, strides, total, psi_k, psi, psi_work,
                      line_in, line_out, 1);

        triad_density_64(total, psi, rho);

        if (M > 0) {
            for (int32_t j = 0; j < M; j++) {
                double od = ou_decay_half[j];
                double od1 = 1.0 - od;
                for (int64_t i = 0; i < total; i++)
                    y[(int64_t)j * total + i] = od * y[(int64_t)j * total + i] + od1 * rho[i];
            }
        }

        for (int64_t i = 0; i < total; i++) {
            double V_mem = 0.0;
            if (M > 0)
                for (int32_t j = 0; j < M; j++)
                    V_mem += eff.lam[j] * y[(int64_t)j * total + i];
            double V_total = V_ext[i] + eff.Lambda * rho[i] + V_mem;
            double angle = -V_total * dt_over_hbar;
            TriadCplx phase = { cos(angle), sin(angle) };
            psi[i] = _cmul(psi[i], phase);
        }

        triad_density_64(total, psi, rho);

        if (M > 0) {
            for (int32_t j = 0; j < M; j++) {
                double od = ou_decay_half[j];
                double od1 = 1.0 - od;
                for (int64_t i = 0; i < total; i++)
                    y[(int64_t)j * total + i] = od * y[(int64_t)j * total + i] + od1 * rho[i];
            }
        }

        {
            double na = noise_amp / sqrt(2.0);
            for (int64_t i = 0; i < total; i++) {
                psi[i].re += na * _randn(&rng_state);
                psi[i].im += na * _randn(&rng_state);
            }
        }

        _fft_nd_apply(N, D, strides, total, psi, psi_k, psi_work,
                      line_in, line_out, 0);
        for (int64_t i = 0; i < total; i++)
            psi_k[i] = _cmul(psi_k[i], half_lin[i]);
        _fft_nd_apply(N, D, strides, total, psi_k, psi, psi_work,
                      line_in, line_out, 1);

        if (bc_mask) {
            for (int64_t i = 0; i < total; i++) {
                psi[i].re *= bc_mask[i];
                psi[i].im *= bc_mask[i];
            }
        }
    }

    triad_density_64(total, psi, rho);

    TriadSolverResultND result;
    result.psi_final = psi;
    result.y_final = y;
    result.density_final = rho;
    result.x = xs;
    result.dx = dx;
    result.peak_t = peak_t;
    result.participation_t = part_t;
    result.n_records = n_rec;

    free(strides);
    free(kvec);
    free(k2);
    free(abs_k);
    free(r2_buf);
    free(V_ext);
    free(half_lin);
    free(lam_buf);
    free(ou_decay_half);
    free(psi_k);
    free(psi_work);
    free(line_in);
    free(line_out);
    free(bc_mask);

    return result;
}

void triad_solver_result_nd_free(TriadSolverResultND *r) {
    if (!r) return;
    free(r->psi_final);
    free(r->y_final);
    free(r->density_final);
    free(r->x);
    free(r->peak_t);
    free(r->participation_t);
}
