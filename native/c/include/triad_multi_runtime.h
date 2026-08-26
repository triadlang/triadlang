#ifndef TRIAD_MULTI_RUNTIME_H
#define TRIAD_MULTI_RUNTIME_H

#include "triad_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TRIAD_COUPLING_DENSITY = 0,
    TRIAD_COUPLING_DC_SUBTRACTED = 1,
    TRIAD_COUPLING_PHASE_COHERENT = 2,
} TriadCouplingMode;

typedef enum {
    TRIAD_ROUTER_PRECISION_WEIGHTED = 0,
    TRIAD_ROUTER_UNIFORM = 1,
} TriadFieldRouterMethod;

typedef double (*TriadKappaModulatorFn)(double *const *rho_snapshots,
                                        int n_subs,
                                        void *user_data);

typedef struct {
    int                    src_id;
    int                    dst_id;
    double                 kappa;
    TriadCouplingMode      mode;
    double                 k_target;


    TriadKappaModulatorFn  kappa_modulator;
    void                  *kappa_modulator_data;
} TriadCouplingLink;

typedef struct {
    int    substrate_id;
    double energy;
    double crystallinity;
    double fdt_precision;
    double k_star;
    double ipr_val;
} TriadSubstrateObservables;

typedef struct {
    TriadFieldRouterMethod method;
    double                 temperature;
} TriadFieldRouter;

typedef struct {
    double             t_start;
    double             t_end;
    TriadCouplingLink *links;
    int                n_links;

    int               *active_ids;
    int                n_active;


    TriadFieldRouter  *router;


    int               *v_ext_override_ids;
    double           **v_ext_overrides;
    int                n_v_ext_overrides;
} TriadSegment;

typedef struct {
    int          id;
    char        *name;
    int          D;

    int          N;
    int64_t      grid_size;
    double       L, dt;
    double       hbar, m, omega;
    double       Lambda, alpha, sigma, Gamma, f_FDT;
    int          fdt_couple;
    double       kT;
    int          M;
    double      *nu;
    double      *lam;
    int          mode;
    uint64_t     seed;
    const char  *V_ext;

    double       dx;
    double      *x;
    double      *V_ext_static;
    TriadCplx   *half_lin;
    double      *ou_decay_half;
    double       noise_amp;

    TriadCplx   *psi;
    double      *y;

    TriadCplx   *psi_scratch;
    TriadCplx   *psi_freq;
    double      *V_couple;
    double      *V_mem;

    double       Lambda_e, alpha_e, Gamma_e, f_FDT_e;
    double      *lam_e;

    int          active;

    int          record_every;
    int          n_records;
    int          cap_records;
    double      *density_traj;
    double      *t_traj;
} TriadSubstrate;

typedef struct {
    double           dt;
    int              record_every;
    double           global_t;
    int              n_substrates;
    int              cap_substrates;
    TriadSubstrate **substrates;
    int              n_segments;
    int              cap_segments;
    TriadSegment    *segments;

    int              diverged;
    int              diverged_segment;
    int              diverged_step;
    char            *diverged_name;
    double           diverged_norm;
} TriadMultiRuntime;

TriadMultiRuntime *triad_mr_new(double dt, int record_every);
void               triad_mr_free(TriadMultiRuntime *rt);

int triad_mr_add_substrate(TriadMultiRuntime *rt,
                           const char *name, int D,
                           int N, double L, double hbar, double m,
                           double omega, double Lambda, double alpha,
                           double sigma, double Gamma, double f_FDT,
                           int fdt_couple, double kT,
                           int M, const double *nu, const double *lam,
                           int p_mode, uint64_t seed,
                           const char *V_ext,
                           const TriadCplx *psi_init);

int triad_mr_add_substrate_1d(TriadMultiRuntime *rt,
                              const char *name,
                              int N, double L, double hbar, double m,
                              double omega, double Lambda, double alpha,
                              double sigma, double Gamma, double f_FDT,
                              int fdt_couple, double kT,
                              int M, const double *nu, const double *lam,
                              int p_mode, uint64_t seed,
                              const char *V_ext,
                              const TriadCplx *psi_init);

int triad_mr_set_v_ext(TriadMultiRuntime *rt, int sid, const double *V);

TriadSubstrate *triad_mr_get(TriadMultiRuntime *rt, int sid);
TriadSubstrate *triad_mr_find(TriadMultiRuntime *rt, const char *name);

double *triad_mr_project_to_shape(const double *src_rho,
                                  int src_D, int src_N,
                                  int dst_D, int dst_N);

void triad_mr_add_segment(TriadMultiRuntime *rt,
                          double t_start, double t_end,
                          TriadCouplingLink *links, int n_links,
                          int *active_ids, int n_active);

void triad_mr_segment_set_router(TriadMultiRuntime *rt,
                                 int seg_index,
                                 TriadFieldRouterMethod method,
                                 double temperature);

int triad_mr_segment_add_v_ext_override(TriadMultiRuntime *rt,
                                        int seg_index,
                                        int substrate_id,
                                        const double *V);

TriadSubstrateObservables triad_mr_compute_observables(const TriadSubstrate *s);

void triad_mr_compute_gates(int n_subs,
                            const TriadSubstrateObservables *obs,
                            const TriadFieldRouter *router,
                            double *gates);

void triad_mr_field_router_route(int n_links,
                                 const TriadCouplingLink *links,
                                 int n_subs,
                                 const double *gates,
                                 double *eff_kappa);

void triad_mr_run(TriadMultiRuntime *rt);

#ifdef __cplusplus
}
#endif

#endif
