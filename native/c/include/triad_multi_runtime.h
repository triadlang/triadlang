#ifndef TRIAD_MULTI_RUNTIME_H
#define TRIAD_MULTI_RUNTIME_H

/* ════════════════════════════════════════════════════════════════════
   TriadLang — Multi-substrate runtime (§11)
   Native C port of runtime/multi_runtime.py.

   This module orchestrates several Triad substrates stepping in
   lockstep with a coupling graph. Each substrate still runs the FULL
   equation P1+P2+P3:

     i ℏ ∂_t Ψ = [-ℏ²/(2m)D² + V_ext + Λ|Ψ|²
                  + V_mem + α(-Δ)^(σ/2) - iΓ] Ψ + η

   where V_ext is augmented by the multi-substrate coupling term

     V_couple_dst(x,t) = Σ_e κ_e * f_e(ρ_src_e(x,t))

   and f_e depends on the coupling mode (density / dc_subtracted).

   Scope of this first port:
     - 1D substrates only (2D/3D will follow once the 1D path is
       byte-stable). Reuses the existing triad_solver 1D scheme exactly:
       split-step Strang with FFT, OU memory, FDT-locked noise.
     - Coupling modes: "density" (default) and "dc_subtracted".
     - active_ids as a static set (callable form requires the
       interpreter — §4 — and lands in a follow-up sub-step).
     - No probes / no Mori-Zwanzig projection yet (those depend on
       observables ND and projections from §10).
   ════════════════════════════════════════════════════════════════════ */

#include "triad_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Coupling modes ─────────────────────────────────────────────── */
typedef enum {
    TRIAD_COUPLING_DENSITY = 0,         /* V_dst += κ · ρ_src                              */
    TRIAD_COUPLING_DC_SUBTRACTED = 1,   /* V_dst += κ · (ρ_src - mean ρ_src)               */
    TRIAD_COUPLING_PHASE_COHERENT = 2,  /* V_dst += κ · Re(Ψ_src · e^{-i k_target · r})    */
} TriadCouplingMode;

/* ── Edge ──────────────────────────────────────────────────────── */
/* k_target is used only by TRIAD_COUPLING_PHASE_COHERENT. In ND it is
   the isotropic scalar applied to r = x [+ y [+ z]] across all axes,
   matching the runtime/multi_runtime.py 2D/3D convention. */
typedef struct {
    int               src_id;
    int               dst_id;
    double            kappa;
    TriadCouplingMode mode;
    double            k_target;
} TriadCouplingEdge;

/* ── Segment ───────────────────────────────────────────────────── */
typedef struct {
    double             t_start;
    double             t_end;
    TriadCouplingEdge *edges;        /* owned */
    int                n_edges;
    /* active set: NULL/0 ⇒ all substrates active; otherwise the
       substrates whose id appears in active_ids[] are active. */
    int               *active_ids;   /* owned, may be NULL */
    int                n_active;
} TriadSegment;

/* ── Substrate ──────────────────────────────────────────────────
   Supports D = 1, 2, 3. Layout for ND fields is row-major:
     1D: idx = i                       size = N
     2D: idx = i*N + j                  size = N²
     3D: idx = i*N² + j*N + k           size = N³
   y has shape (M, grid_size) flat: y[(j)*grid_size + idx].
   ─────────────────────────────────────────────────────────────── */
typedef struct {
    int          id;
    char        *name;          /* owned */
    int          D;             /* 1, 2, or 3 */
    /* physical parameters — mirrors TriadSolverC fields used per-step */
    int          N;             /* points per axis */
    int64_t      grid_size;     /* N, N², or N³ */
    double       L, dt;
    double       hbar, m, omega;
    double       Lambda, alpha, sigma, Gamma, f_FDT;
    int          M;             /* memory chain length */
    double      *nu;            /* M, owned */
    double      *lam;           /* M, owned */
    int          mode;          /* 0=linear, 1=thermal, 2=full (P1+P2+P3) */
    uint64_t     seed;
    const char  *V_ext;         /* NULL or "harmonic"; static_V_ext[] is built */

    /* derived per-step state */
    double       dx;
    double      *x;             /* N (axis coords; same on every axis) */
    double      *V_ext_static;  /* grid_size */
    TriadCplx   *half_lin;      /* grid_size — exp(-i H_lin |k|² dt/(2ℏ) - Γ dt/(2ℏ)) */
    double      *ou_decay_half; /* M — exp(-ν dt / 2) */
    double       noise_amp;     /* sqrt(f_FDT dt / dx^D), 0 if f_FDT==0 */
    /* dynamic state */
    TriadCplx   *psi;           /* grid_size */
    double      *y;             /* M × grid_size (row major) */
    /* per-step scratch */
    TriadCplx   *psi_scratch;   /* grid_size */
    TriadCplx   *psi_freq;      /* grid_size */
    double      *V_couple;      /* grid_size */
    double      *V_mem;         /* grid_size */
    /* effective values (linear/thermal/full mode) */
    double       Lambda_e, alpha_e, Gamma_e, f_FDT_e;
    double      *lam_e;         /* M */
    /* lifecycle flags */
    int          active;        /* 1 = stepped this segment, 0 = frozen */
    /* trajectory recording (refilled by run()) */
    int          record_every;
    int          n_records;     /* current count */
    int          cap_records;   /* allocated capacity */
    double      *density_traj;  /* grid_size × n_records (flat: [k*grid_size + i]) */
    double      *t_traj;        /* cap_records */
} TriadSubstrate;

/* ── MultiRuntime ──────────────────────────────────────────────── */
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
    /* divergence detector state */
    int              diverged;
    int              diverged_segment;
    int              diverged_step;
    char            *diverged_name;
    double           diverged_norm;
} TriadMultiRuntime;

/* ── Construction / lifecycle ──────────────────────────────────── */
TriadMultiRuntime *triad_mr_new(double dt, int record_every);
void               triad_mr_free(TriadMultiRuntime *rt);

/* Adds a substrate with the given parameters. `psi_init` may be NULL,
 * in which case a default Gaussian is installed (mirrors Python). The
 * runtime takes a copy. `D` selects 1, 2, or 3 dimensions.
 * psi_init layout:
 *   1D: N elements
 *   2D: N² elements, row-major [i*N + j]
 *   3D: N³ elements, row-major [i*N² + j*N + k]
 * Returns the assigned id (>= 0) or -1 on error. */
int triad_mr_add_substrate(TriadMultiRuntime *rt,
                           const char *name, int D,
                           int N, double L, double hbar, double m,
                           double omega, double Lambda, double alpha,
                           double sigma, double Gamma, double f_FDT,
                           int M, const double *nu, const double *lam,
                           int p_mode, uint64_t seed,
                           const char *V_ext,
                           const TriadCplx *psi_init);

/* 1D convenience wrapper preserved for backward compatibility. */
int triad_mr_add_substrate_1d(TriadMultiRuntime *rt,
                              const char *name,
                              int N, double L, double hbar, double m,
                              double omega, double Lambda, double alpha,
                              double sigma, double Gamma, double f_FDT,
                              int M, const double *nu, const double *lam,
                              int p_mode, uint64_t seed,
                              const char *V_ext,
                              const TriadCplx *psi_init);

/* Project a source density of one dimensionality onto a destination grid
 * shape. Mirrors runtime/multi_runtime.py:_project_to_shape exactly:
 *   - same D: returns a copy of the source.
 *   - 3D src ↔ 1D/2D dst: sum-projection over orthogonal axes.
 *   - 1D src ↔ 2D/3D dst: outer product with a normalised Gaussian
 *     envelope (width = N/8) along the missing axes.
 *   - unsupported: returns zeros.
 * Caller owns the returned buffer (size = N_dst^D_dst), frees with free(). */
double *triad_mr_project_to_shape(const double *src_rho,
                                  int src_D, int src_N,
                                  int dst_D, int dst_N);

/* Append a segment. Takes ownership of `edges` and `active_ids`. */
void triad_mr_add_segment(TriadMultiRuntime *rt,
                          double t_start, double t_end,
                          TriadCouplingEdge *edges, int n_edges,
                          int *active_ids, int n_active);

/* Run the schedule. Sets rt->diverged on divergence. */
void triad_mr_run(TriadMultiRuntime *rt);

/* Accessors. */
TriadSubstrate *triad_mr_get(TriadMultiRuntime *rt, int sid);
TriadSubstrate *triad_mr_find(TriadMultiRuntime *rt, const char *name);

#ifdef __cplusplus
}
#endif

#endif /* TRIAD_MULTI_RUNTIME_H */
