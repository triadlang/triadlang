#define _POSIX_C_SOURCE 200809L
#include "triad_compiler.h"
#include "triad_format.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *_dup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *out = (char *)malloc(n);
    if (out) memcpy(out, s, n);
    return out;
}

static double *_dup_doubles(const double *src, int n) {
    if (n <= 0) return NULL;
    double *out = (double *)malloc(sizeof(double) * (size_t)n);
    memcpy(out, src, sizeof(double) * (size_t)n);
    return out;
}

TriadReadoutConfig *triad_readout_config_new_default(void) {
    TriadReadoutConfig *r = (TriadReadoutConfig *)calloc(1, sizeof(*r));
    r->what = _dup("triad");
    static const char *defaults[] = {
        "crystallinity", "k_star", "peak", "ipr",
        "participation", "fwhm", "norm", "density_pca"
    };
    int n = (int)(sizeof(defaults) / sizeof(defaults[0]));
    r->fields = (char **)malloc(sizeof(char *) * (size_t)n);
    for (int i = 0; i < n; ++i) r->fields[i] = _dup(defaults[i]);
    r->n_fields = n;
    return r;
}

void triad_readout_config_free(TriadReadoutConfig *r) {
    if (!r) return;
    free(r->what);
    for (int i = 0; i < r->n_fields; ++i) free(r->fields[i]);
    free(r->fields);
    free(r);
}

static void _init_substrate_defaults(TriadSubstrateConfig *s) {
    s->D = 1;
    s->N = 128;
    s->L = 32.0;
    s->Lambda = -0.5;
    s->alpha = 0.15;
    s->sigma = 1.5;
    s->Gamma = 0.05;
    s->f_FDT = 0.002;
    s->fdt_couple = 1;
    s->kT = 1.0;
    s->omega = 0.05;
    s->V_ext = _dup("harmonic");
    s->regime = _dup("B0");
    s->name = _dup("");
    static const double default_nu[]  = { 2.0, 0.5, 0.1 };
    static const double default_lam[] = { -0.3, -0.2, -0.1 };
    s->M = 3;
    s->nu  = _dup_doubles(default_nu, 3);
    s->lam = _dup_doubles(default_lam, 3);
}

static void _substrate_free_contents(TriadSubstrateConfig *s) {
    free(s->name); free(s->V_ext); free(s->regime);
    free(s->nu); free(s->lam);
}

static void _coupling_free_contents(TriadCouplingConfig *c) {
    free(c->src); free(c->dst); free(c->mode);
}

void triad_compiled_program_free(TriadCompiledProgram *prog) {
    if (!prog) return;
    for (int i = 0; i < prog->n_substrates; ++i) {
        _substrate_free_contents(&prog->substrates[i]);
    }
    free(prog->substrates);
    for (int i = 0; i < prog->n_couplings; ++i) {
        _coupling_free_contents(&prog->couplings[i]);
    }
    free(prog->couplings);
    triad_readout_config_free(prog->readout);
    free(prog->backend);
    for (int i = 0; i < prog->n_meta; ++i) {
        free(prog->meta_keys[i]);
        free(prog->meta_vals[i]);
    }
    free(prog->meta_keys);
    free(prog->meta_vals);
    free(prog);
}

static TriadCompiledProgram *_new_program(void) {
    TriadCompiledProgram *p = (TriadCompiledProgram *)calloc(1, sizeof(*p));
    p->T = 5.0;
    p->dt = 0.005;
    p->backend = _dup("auto");
    p->readout = triad_readout_config_new_default();
    return p;
}

static void _push_meta(TriadCompiledProgram *p, const char *k, const char *v) {
    int n = p->n_meta + 1;
    p->meta_keys = (char **)realloc(p->meta_keys, sizeof(char *) * (size_t)n);
    p->meta_vals = (char **)realloc(p->meta_vals, sizeof(char *) * (size_t)n);
    p->meta_keys[p->n_meta] = _dup(k);
    p->meta_vals[p->n_meta] = _dup(v);
    p->n_meta = n;
}

static void _push_meta_int(TriadCompiledProgram *p, const char *k, int v) {
    char buf[32]; snprintf(buf, sizeof buf, "%d", v);
    _push_meta(p, k, buf);
}

static void _push_meta_tuple_double(TriadCompiledProgram *p, const char *k,
                                    const double *vs, int n) {

    char buf[256]; size_t off = 0;
    off += (size_t)snprintf(buf + off, sizeof buf - off, "(");
    for (int i = 0; i < n; ++i) {
        char num[40]; triad_py_repr_float(vs[i], num, sizeof num);
        if (i + 1 < n) {
            off += (size_t)snprintf(buf + off, sizeof buf - off, "%s, ", num);
        } else {
            off += (size_t)snprintf(buf + off, sizeof buf - off,
                                    (n == 1) ? "%s," : "%s", num);
        }
    }
    snprintf(buf + off, sizeof buf - off, ")");
    _push_meta(p, k, buf);
}

static TriadSubstrateConfig *_alloc_substrates(int count) {
    TriadSubstrateConfig *arr = (TriadSubstrateConfig *)calloc(
        (size_t)count, sizeof(TriadSubstrateConfig));
    for (int i = 0; i < count; ++i) _init_substrate_defaults(&arr[i]);
    return arr;
}

static TriadCouplingConfig *_alloc_couplings(int count) {
    return (TriadCouplingConfig *)calloc((size_t)count, sizeof(TriadCouplingConfig));
}

TriadCompiledProgram *triad_compile_classify(int n_classes,
                                             int n_features,
                                             int depth) {
    if (n_classes <= 0) n_classes = 2;
    if (n_features <= 0) n_features = 128;
    if (depth <= 0) depth = 4;

    int log_term = 0;

    int v = n_classes + 1;
    while ((1 << log_term) < v) log_term++;

    int n_subs = depth > log_term ? depth : log_term;

    TriadCompiledProgram *p = _new_program();
    p->substrates = _alloc_substrates(n_subs);
    p->n_substrates = n_subs;

    for (int i = 0; i < n_subs; ++i) {
        TriadSubstrateConfig *s = &p->substrates[i];
        free(s->name);
        char name[32]; snprintf(name, sizeof name, "cls_%d", i);
        s->name = _dup(name);
        s->N = n_features;
        s->Lambda = -0.5 * (1.0 + 0.2 * (double)i);
        s->alpha = 0.15;

        double nu_vals[3]  = { 2.0, 0.5 / (double)(i + 1), 0.1 / (double)(i + 1) };
        double lam_scale = 1.0 / (double)(i + 1);
        double lam_vals[3] = {
            -0.3 * lam_scale, -0.2 * lam_scale, -0.1 * lam_scale
        };
        free(s->nu); free(s->lam);
        s->nu  = _dup_doubles(nu_vals, 3);
        s->lam = _dup_doubles(lam_vals, 3);
        s->M = 3;
        s->omega = 0.05 * (1.0 + 0.1 * (double)i);
    }

    int n_links = (n_subs - 1) + 1;
    if (n_subs == 1) n_links = 1;
    p->couplings = _alloc_couplings(n_links);
    int e = 0;
    for (int i = 0; i < n_subs - 1; ++i) {
        char a[32], b[32];
        snprintf(a, sizeof a, "cls_%d", i);
        snprintf(b, sizeof b, "cls_%d", i + 1);
        p->couplings[e].src = _dup(a);
        p->couplings[e].dst = _dup(b);
        p->couplings[e].kappa = -3.0;
        p->couplings[e].mode = _dup("density");
        e++;
    }
    {
        char a[32], b[32];
        snprintf(a, sizeof a, "cls_%d", n_subs - 1);
        snprintf(b, sizeof b, "cls_0");
        p->couplings[e].src = _dup(a);
        p->couplings[e].dst = _dup(b);
        p->couplings[e].kappa = -1.5;
        p->couplings[e].mode = _dup("density");
        e++;
    }
    p->n_couplings = e;

    _push_meta(p, "task", "classify");
    _push_meta_int(p, "n_classes", n_classes);
    return p;
}

TriadCompiledProgram *triad_compile_generate(const char *pattern_type, int N) {
    if (!pattern_type) pattern_type = "crystal";
    if (N <= 0) N = 128;

    double lam_split_crystal[3]  = { 1.125, 0.375, 0.125 };
    double nu_crystal[3]         = { 10.0, 0.5, 0.1 };
    double lam_split_filament[3] = { -0.3, -0.2, -0.1 };
    double nu_filament[3]        = { 2.0, 0.5, 0.1 };
    double lam_split_lattice[3]  = { 0.75, 0.25, 0.075 };
    double nu_lattice[3]         = { 10.0, 0.5, 0.1 };

    const double *nu_p = NULL, *lam_p = NULL;
    int M = 0;
    if (strcmp(pattern_type, "crystal") == 0) {
        nu_p = nu_crystal; lam_p = lam_split_crystal; M = 3;
    } else if (strcmp(pattern_type, "filament") == 0) {
        nu_p = nu_filament; lam_p = lam_split_filament; M = 3;
    } else if (strcmp(pattern_type, "lattice") == 0) {
        nu_p = nu_lattice; lam_p = lam_split_lattice; M = 3;
    } else {
        nu_p = nu_filament; lam_p = lam_split_filament; M = 3;
    }

    TriadCompiledProgram *p = _new_program();
    p->substrates = _alloc_substrates(1);
    p->n_substrates = 1;
    TriadSubstrateConfig *s = &p->substrates[0];
    free(s->name); s->name = _dup("gen_0");
    s->N = N;
    s->Lambda = (strcmp(pattern_type, "crystal") == 0) ? -8.0 : -0.5;
    s->alpha  = (strcmp(pattern_type, "crystal") == 0) ?  0.1 : 0.15;
    s->sigma  = (strcmp(pattern_type, "crystal") == 0) ?  1.8 : 1.5;
    free(s->nu); free(s->lam);
    s->nu  = _dup_doubles(nu_p, M);
    s->lam = _dup_doubles(lam_p, M);
    s->M = M;
    free(s->V_ext); s->V_ext = NULL;

    p->T = (strcmp(pattern_type, "crystal") == 0) ? 15.0 : 10.0;

    _push_meta(p, "task", "generate");
    _push_meta(p, "pattern_type", pattern_type);
    return p;
}

TriadCompiledProgram *triad_compile_remember(const double *timescales,
                                             int n_timescales, int N) {

    double default_ts[3] = { 1.0, 5.0, 20.0 };
    if (!timescales || n_timescales <= 0) {
        timescales = default_ts;
        n_timescales = 3;
    }
    if (N <= 0) N = 128;

    double *nu  = (double *)malloc(sizeof(double) * (size_t)n_timescales);
    double *lam = (double *)malloc(sizeof(double) * (size_t)n_timescales);
    for (int i = 0; i < n_timescales; ++i) {
        nu[i]  = 1.0 / timescales[i];
        lam[i] = -0.3 / (double)(i + 1);
    }

    TriadCompiledProgram *p = _new_program();
    p->substrates = _alloc_substrates(1);
    p->n_substrates = 1;
    TriadSubstrateConfig *s = &p->substrates[0];
    free(s->name); s->name = _dup("mem_0");
    s->N = N;
    s->Lambda = -0.3;
    s->Gamma = 0.02;
    free(s->nu); free(s->lam);
    s->nu = nu; s->lam = lam;
    s->M = n_timescales;

    _push_meta(p, "task", "remember");
    _push_meta_tuple_double(p, "timescales", timescales, n_timescales);
    return p;
}

TriadCompiledProgram *triad_compile_couple(int n_substrates,
                                           const char *topology,
                                           double kappa) {
    if (n_substrates <= 0) n_substrates = 3;
    if (!topology) topology = "ring";
    if (kappa == 0.0) kappa = -3.0;

    TriadCompiledProgram *p = _new_program();
    p->substrates = _alloc_substrates(n_substrates);
    p->n_substrates = n_substrates;
    for (int i = 0; i < n_substrates; ++i) {
        TriadSubstrateConfig *s = &p->substrates[i];
        free(s->name);
        char nm[32]; snprintf(nm, sizeof nm, "s%d", i);
        s->name = _dup(nm);

    }

    int max_links = 0;
    if (strcmp(topology, "ring") == 0)      max_links = n_substrates;
    else if (strcmp(topology, "triad") == 0) max_links = n_substrates * (n_substrates - 1);
    else if (strcmp(topology, "star") == 0) max_links = 2 * (n_substrates - 1);

    p->couplings = _alloc_couplings(max_links > 0 ? max_links : 1);
    int e = 0;
    if (strcmp(topology, "ring") == 0) {
        for (int i = 0; i < n_substrates; ++i) {
            char a[32], b[32];
            snprintf(a, sizeof a, "s%d", i);
            snprintf(b, sizeof b, "s%d", (i + 1) % n_substrates);
            p->couplings[e].src = _dup(a);
            p->couplings[e].dst = _dup(b);
            p->couplings[e].kappa = kappa;
            p->couplings[e].mode = _dup("density");
            e++;
        }
    } else if (strcmp(topology, "triad") == 0) {
        for (int i = 0; i < n_substrates; ++i) {
            for (int j = 0; j < n_substrates; ++j) {
                if (i == j) continue;
                char a[32], b[32];
                snprintf(a, sizeof a, "s%d", i);
                snprintf(b, sizeof b, "s%d", j);
                p->couplings[e].src = _dup(a);
                p->couplings[e].dst = _dup(b);
                p->couplings[e].kappa = kappa / (double)n_substrates;
                p->couplings[e].mode = _dup("density");
                e++;
            }
        }
    } else if (strcmp(topology, "star") == 0) {
        for (int i = 1; i < n_substrates; ++i) {
            char a[32], b[32];
            snprintf(a, sizeof a, "s%d", i);
            snprintf(b, sizeof b, "s0");
            p->couplings[e].src = _dup(a);
            p->couplings[e].dst = _dup(b);
            p->couplings[e].kappa = kappa;
            p->couplings[e].mode = _dup("density");
            e++;
            snprintf(a, sizeof a, "s0");
            snprintf(b, sizeof b, "s%d", i);
            p->couplings[e].src = _dup(a);
            p->couplings[e].dst = _dup(b);
            p->couplings[e].kappa = kappa;
            p->couplings[e].mode = _dup("density");
            e++;
        }
    }
    p->n_couplings = e;

    _push_meta(p, "task", "couple");
    _push_meta(p, "topology", topology);
    return p;
}

TriadCompiledProgram *triad_compile_sat(int n_vars, int n_clauses) {
    if (n_vars <= 0) n_vars = 3;
    if (n_clauses <= 0) n_clauses = 9;

    int N = n_vars * 8;
    if (N < 64) N = 64;

    TriadCompiledProgram *p = _new_program();
    p->substrates = _alloc_substrates(1);
    p->n_substrates = 1;
    TriadSubstrateConfig *s = &p->substrates[0];
    free(s->name); s->name = _dup("sat_0");
    s->N = N;
    s->Lambda = -5.0;
    s->alpha = 0.1;
    s->sigma = 1.7;
    free(s->V_ext); s->V_ext = NULL;
    free(s->nu); free(s->lam);
    double sat_nu[3]  = { 2.0, 0.5, 0.1 };
    double sat_lam[3] = { -0.5, -0.3, -0.1 };
    s->nu = _dup_doubles(sat_nu, 3);
    s->lam = _dup_doubles(sat_lam, 3);
    s->M = 3;

    p->T = 10.0;

    _push_meta(p, "task", "sat");
    _push_meta_int(p, "n_vars", n_vars);
    _push_meta_int(p, "n_clauses", n_clauses);
    return p;
}

const char *triad_topology_from_couplings(int n_couplings, int n_subs) {
    if (n_couplings == n_subs) return "ring";
    if (n_couplings == n_subs * (n_subs - 1)) return "triad";
    if (n_couplings == 0) return "ring";
    return "ring";
}

static int _find_substrate_id(TriadMultiRuntime *rt, const char *name) {
    TriadSubstrate *s = triad_mr_find(rt, name);
    return s ? s->id : -1;
}

static TriadCouplingMode _mode_from_string(const char *mode) {
    if (!mode) return TRIAD_COUPLING_DENSITY;
    if (strcmp(mode, "dc_subtracted") == 0)  return TRIAD_COUPLING_DC_SUBTRACTED;
    if (strcmp(mode, "phase_coherent") == 0) return TRIAD_COUPLING_PHASE_COHERENT;
    return TRIAD_COUPLING_DENSITY;
}

TriadMultiRuntime *triad_compiler_instantiate(const TriadCompiledProgram *prog,
                                              uint64_t seed) {
    if (!prog) return NULL;
    TriadMultiRuntime *rt = triad_mr_new(prog->dt, 4);

    int mode_triad = 2;

    for (int i = 0; i < prog->n_substrates; ++i) {
        const TriadSubstrateConfig *s = &prog->substrates[i];

        triad_mr_add_substrate(rt, s->name, s->D,
                               s->N, s->L,
                               1.0, 1.0, s->omega,
                               s->Lambda, s->alpha, s->sigma,
                               s->Gamma, s->f_FDT,
                               s->fdt_couple, s->kT,
                               s->M, s->nu, s->lam,
                               mode_triad,
                               seed + (uint64_t)i,
                               s->V_ext,
                               NULL );
    }

    int ne = prog->n_couplings;
    TriadCouplingLink *links = NULL;
    if (ne > 0) {
        links = (TriadCouplingLink *)malloc(sizeof(TriadCouplingLink) * (size_t)ne);
        for (int e = 0; e < ne; ++e) {
            const TriadCouplingConfig *c = &prog->couplings[e];
            int sid = _find_substrate_id(rt, c->src);
            int did = _find_substrate_id(rt, c->dst);
            links[e] = (TriadCouplingLink){
                .src_id = sid, .dst_id = did,
                .kappa = c->kappa,
                .mode = _mode_from_string(c->mode),
                .k_target = c->k_target,
            };
        }
    }
    triad_mr_add_segment(rt, 0.0, prog->T, links, ne, NULL, 0);
    return rt;
}
