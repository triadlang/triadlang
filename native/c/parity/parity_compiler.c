/*
 * parity_compiler.c — Native fingerprint of the physical compiler,
 * compared against scripts/compiler_to_text.py.
 *
 * P1+P2+P3 invariance: every compiled program ships with non-zero
 * Λ, α, Γ, f_FDT, and a memory chain — the three pillars are
 * constitutive of the defaults, not optional. The bias IEEE-754
 * values (e.g. omega = 0.055000000000000001) are reproduced bit-for-bit
 * because we use the same shortest-roundtrip formatter the Python
 * repr() uses (triad_py_repr_float).
 */
#include "triad_compiler.h"
#include "triad_format.h"
#include "triad_rt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void _fmt(double v, char *out, size_t n) {
    triad_py_repr_float(v, out, n);
}

static void _emit_tuple(const double *xs, int n) {
    putchar('(');
    char buf[64];
    for (int i = 0; i < n; ++i) {
        _fmt(xs[i], buf, sizeof buf);
        fputs(buf, stdout);
        if (i + 1 < n) fputs(", ", stdout);
    }
    if (n == 1) fputc(',', stdout);
    putchar(')');
}

/* Lexicographic key compare for metadata sort. */
static int _cmp_strs(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static void dump_program(const char *name, const TriadCompiledProgram *p) {
    printf("fixture %s\n", name);
    char tb[64], db[64];
    _fmt(p->T, tb, sizeof tb);
    _fmt(p->dt, db, sizeof db);
    printf("T %s dt %s backend %s n_subs %d n_couplings %d\n",
           tb, db, p->backend, p->n_substrates, p->n_couplings);

    /* meta: sort keys lexicographically (matches sorted(prog.metadata)) */
    if (p->n_meta == 0) {
        printf("meta\n");
    } else {
        char **keys = (char **)malloc(sizeof(char *) * (size_t)p->n_meta);
        for (int i = 0; i < p->n_meta; ++i) keys[i] = p->meta_keys[i];
        qsort(keys, (size_t)p->n_meta, sizeof(char *), _cmp_strs);
        printf("meta");
        for (int i = 0; i < p->n_meta; ++i) {
            /* find value for this sorted key */
            const char *v = NULL;
            for (int j = 0; j < p->n_meta; ++j) {
                if (strcmp(p->meta_keys[j], keys[i]) == 0) {
                    v = p->meta_vals[j]; break;
                }
            }
            printf(" %s=%s", keys[i], v ? v : "");
        }
        putchar('\n');
        free(keys);
    }

    /* readout */
    printf("readout what=%s fields=(", p->readout->what);
    for (int i = 0; i < p->readout->n_fields; ++i) {
        fputs(p->readout->fields[i], stdout);
        if (i + 1 < p->readout->n_fields) fputc(',', stdout);
    }
    puts(")");

    /* substrates */
    for (int i = 0; i < p->n_substrates; ++i) {
        const TriadSubstrateConfig *s = &p->substrates[i];
        char lb[64], ab[64], sb[64], gb[64], fb[64], wb[64], xb[64];
        _fmt(s->L, lb, sizeof lb);
        _fmt(s->Lambda, ab, sizeof ab);
        _fmt(s->alpha, sb, sizeof sb);
        _fmt(s->sigma, gb, sizeof gb);
        _fmt(s->Gamma, fb, sizeof fb);
        _fmt(s->f_FDT, wb, sizeof wb);
        _fmt(s->omega, xb, sizeof xb);
        printf("sub %d name=%s D=1 N=%d L=%s "
               "Lambda=%s alpha=%s sigma=%s Gamma=%s f_FDT=%s omega=%s\n",
               i, s->name, s->N, lb, ab, sb, gb, fb, wb, xb);
        printf("sub %d nu=", i); _emit_tuple(s->nu, s->M);
        printf(" lam="); _emit_tuple(s->lam, s->M);
        printf(" V_ext=%s regime=%s\n",
               s->V_ext ? s->V_ext : "None", s->regime);
    }

    /* couplings */
    for (int j = 0; j < p->n_couplings; ++j) {
        const TriadCouplingConfig *c = &p->couplings[j];
        char kb[64];
        _fmt(c->kappa, kb, sizeof kb);
        printf("couple %d src=%s dst=%s kappa=%s mode=%s\n",
               j, c->src, c->dst, kb, c->mode);
    }
}

int main(void) {
    TriadCompiledProgram *p;

    /* classify */
    p = triad_compile_classify(-1, -1, -1);
    dump_program("classify_default", p);
    triad_compiled_program_free(p);

    p = triad_compile_classify(5, 64, 6);
    dump_program("classify_5_classes_depth_6", p);
    triad_compiled_program_free(p);

    p = triad_compile_classify(2, 32, 3);
    dump_program("classify_2_classes", p);
    triad_compiled_program_free(p);

    /* generate */
    p = triad_compile_generate("crystal", 128);
    dump_program("generate_crystal", p);
    triad_compiled_program_free(p);

    p = triad_compile_generate("filament", 64);
    dump_program("generate_filament", p);
    triad_compiled_program_free(p);

    p = triad_compile_generate("lattice", 64);
    dump_program("generate_lattice", p);
    triad_compiled_program_free(p);

    /* remember */
    p = triad_compile_remember(NULL, 0, -1);
    dump_program("remember_default", p);
    triad_compiled_program_free(p);
    {
        double ts[4] = { 0.5, 2.0, 8.0, 32.0 };
        p = triad_compile_remember(ts, 4, 64);
        dump_program("remember_4ts", p);
        triad_compiled_program_free(p);
    }

    /* couple */
    p = triad_compile_couple(4, "ring", -3.0);
    dump_program("couple_ring", p);
    triad_compiled_program_free(p);
    p = triad_compile_couple(3, "full", -3.0);
    dump_program("couple_full", p);
    triad_compiled_program_free(p);
    p = triad_compile_couple(4, "star", -3.0);
    dump_program("couple_star", p);
    triad_compiled_program_free(p);

    /* sat */
    p = triad_compile_sat(-1, -1);
    dump_program("sat_default", p);
    triad_compiled_program_free(p);
    p = triad_compile_sat(5, 20);
    dump_program("sat_5vars", p);
    triad_compiled_program_free(p);

    return 0;
}
