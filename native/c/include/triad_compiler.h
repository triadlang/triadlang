#ifndef TRIAD_COMPILER_H
#define TRIAD_COMPILER_H

#include "triad_rt.h"
#include "triad_multi_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char  *what;
    char **fields;
    int    n_fields;
} TriadReadoutConfig;

TriadReadoutConfig *triad_readout_config_new_default(void);
void                triad_readout_config_free(TriadReadoutConfig *r);

typedef struct {
    char    *name;
    int      D;
    int      N;
    double   L;
    double   Lambda;
    double   alpha;
    double   sigma;
    double   Gamma;
    double   f_FDT;
    int      fdt_couple;
    double   kT;
    int      M;
    double  *nu;
    double  *lam;
    char    *V_ext;
    double   omega;
    char    *regime;
} TriadSubstrateConfig;

typedef struct {
    char  *src;
    char  *dst;
    double kappa;
    char  *mode;
    double k_target;
} TriadCouplingConfig;

typedef struct {
    TriadSubstrateConfig *substrates;
    int                   n_substrates;
    TriadCouplingConfig  *couplings;
    int                   n_couplings;
    TriadReadoutConfig   *readout;
    double                T;
    double                dt;
    char                 *backend;

    char                **meta_keys;
    char                **meta_vals;
    int                   n_meta;
} TriadCompiledProgram;

void triad_compiled_program_free(TriadCompiledProgram *prog);

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

const char *triad_topology_from_couplings(int n_couplings, int n_substrates);

TriadMultiRuntime *triad_compiler_instantiate(const TriadCompiledProgram *prog,
                                              uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif
