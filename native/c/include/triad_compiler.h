#ifndef TRIAD_COMPILER_H
#define TRIAD_COMPILER_H

/* ════════════════════════════════════════════════════════════════════
   TriadLang — Physical compiler (§5)
   Native C port of runtime/compiler.py.

   Transforms a task specification into a declarative CompiledProgram:
     list of SubstrateConfig (each with P1+P2+P3 parameters: Λ, α, σ,
       Γ, f_FDT, ν[], λ[], V_ext, regime, ω)
     list of CouplingConfig (src→dst with κ and mode)
     ReadoutConfig
     T, dt

   IMPORTANT — equation rules respected:
     - The compiler does NOT impose metrics on the field. It only writes
       configurations; the equation rules what emerges at runtime.
     - The three pillars (P1: dispersion/FFT, P2: memory Λ|Ψ|² + V_mem,
       P3: dissipation/FDT noise) are ALL present in every config. Tasks
       differ only in calibration (the Λ, ν, λ profiles), not by zeroing
       pillars.
     - The arithmetic is NOT standard: ν=(2.0, 0.5/(i+1), 0.1/(i+1)),
       λ_scale = 1/(i+1), Λ profiles per pattern_type — all ported
       BYTE-IDENTICAL to runtime/compiler.py, no "clean-up" of values.

   This module is a pure-data layer. The instantiation into a runnable
   TriadMultiRuntime lives at the bottom of this header; it is the
   bridge between §5 (compiler) and §5 multi-runtime (already done).
   ════════════════════════════════════════════════════════════════════ */

#include "triad_rt.h"
#include "triad_multi_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Readout config (subset of runtime/equation_runtime.py) ───── */

typedef struct {
    char  *what;                    /* "full" | "summary" | ... — owned */
    char **fields;                  /* tuple of field names — owned */
    int    n_fields;
} TriadReadoutConfig;

TriadReadoutConfig *triad_readout_config_new_default(void);
void                triad_readout_config_free(TriadReadoutConfig *r);

/* ── Substrate config ──────────────────────────────────────────── */

typedef struct {
    char    *name;          /* owned */
    int      D;             /* 1, 2, 3 — defaults to 1 */
    int      N;
    double   L;
    double   Lambda;
    double   alpha;
    double   sigma;
    double   Gamma;
    double   f_FDT;
    int      M;             /* len(nu) == len(lam) */
    double  *nu;            /* owned, M */
    double  *lam;           /* owned, M */
    char    *V_ext;         /* "harmonic" | NULL | ... — owned (may be NULL) */
    double   omega;
    char    *regime;        /* "B0" | ... — owned */
} TriadSubstrateConfig;

/* ── Coupling config ────────────────────────────────────────── */

typedef struct {
    char  *src;             /* substrate name — owned */
    char  *dst;             /* substrate name — owned */
    double kappa;
    char  *mode;            /* "density" | "dc_subtracted" | "phase_coherent" — owned */
    double k_target;        /* only used for phase_coherent; 0 otherwise */
} TriadCouplingConfig;

/* ── Compiled program ──────────────────────────────────────────── */

typedef struct {
    TriadSubstrateConfig *substrates;  /* owned, n_substrates */
    int                   n_substrates;
    TriadCouplingConfig  *couplings;   /* owned, n_couplings */
    int                   n_couplings;
    TriadReadoutConfig   *readout;     /* owned */
    double                T;
    double                dt;
    char                 *backend;     /* "auto" | "cpu" | ... — owned */
    /* metadata as key/value string pairs (owned) */
    char                **meta_keys;
    char                **meta_vals;
    int                   n_meta;
} TriadCompiledProgram;

void triad_compiled_program_free(TriadCompiledProgram *prog);

/* ── Compiler entry points (one per task) ──────────────────────── */

/* All compilers return owned objects; caller frees with
   triad_compiled_program_free(). Parameters mirror runtime/compiler.py
   defaults exactly. Pass <=0 to use the Python default for any field. */

TriadCompiledProgram *triad_compile_classify(int n_classes,
                                             int n_features,
                                             int depth);

TriadCompiledProgram *triad_compile_generate(const char *pattern_type,
                                             int N);

TriadCompiledProgram *triad_compile_remember(const double *timescales,
                                             int n_timescales,
                                             int N);

TriadCompiledProgram *triad_compile_couple(int n_substrates,
                                           const char *topology,
                                           double kappa);

TriadCompiledProgram *triad_compile_sat(int n_vars, int n_clauses);

/* Helper: derive topology label from coupling count (matches
   runtime/compiler.py:_topology_from_couplings exactly). */
const char *triad_topology_from_couplings(int n_couplings, int n_substrates);

/* ── Instantiation (bridge to TriadMultiRuntime) ───────────────── */

/* Build a TriadMultiRuntime from the compiled program with all three
   pillars live (P1+P2+P3): every substrate gets the full
     Λ|Ψ|² + V_ext + Σ λ_j y_j + α(-Δ)^{σ/2} - iΓ
   plus the FDT-locked noise term. Coupling edges from the program's
   couplings are added to a single segment of duration `T`. The default
   ψ₀ is a centred Gaussian (the runtime's own default when psi_init
   is NULL).

   This is the C analogue of compiler.py:instantiate(); it differs only
   in returning a TriadMultiRuntime directly (which already exists and
   is byte-validated) instead of an EquationRuntime (still §7 missing).
   The equation rules; the runtime executes. */
TriadMultiRuntime *triad_compiler_instantiate(const TriadCompiledProgram *prog,
                                              uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* TRIAD_COMPILER_H */
