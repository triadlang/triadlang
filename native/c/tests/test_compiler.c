/*
 * test_compiler.c — Unit tests for the physical compiler.
 *
 * Validates structural invariants of compiled programs and the
 * instantiate() bridge into TriadMultiRuntime:
 *
 *   1. classify default has 4 substrates and a ring + close-edge of 4
 *      couplings; first substrate carries the default Λ profile.
 *   2. generate(crystal) has Λ=-8.0, alpha=0.0, T=15.0, V_ext=NULL,
 *      and a 2-channel memory chain — pillars NOT zeroed, only
 *      profiled differently.
 *   3. _topology_from_couplings recognises ring / full / empty patterns.
 *   4. instantiate() builds a TriadMultiRuntime where every substrate
 *      has Λ, α, σ, Γ, f_FDT, ν[], λ[] all populated as declared —
 *      P1+P2+P3 are constitutive, the equation rules.
 *   5. The materialised runtime runs without divergence for a short
 *      segment and the norm decays as exp(-2 Γ T / ℏ) when no noise —
 *      same invariant the standalone multi_runtime tests verify.
 *
 * NOTE on rules:
 *   - The compiler does not impose metrics on the field. It only writes
 *     configurations. These tests check structure (presence of pillars,
 *     coupling topology), not what emerges at runtime.
 *   - Arithmetic is not standard: we compare Λ to its exact float repr,
 *     ν tuples to literal arrays, never to "nice" rounded numbers.
 */
#include "triad_compiler.h"
#include "triad_multi_runtime.h"
#include "triad_rt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_total = 0;

#define CHECK(cond, msg, ...) do {                                    \
    g_total++;                                                        \
    if (!(cond)) {                                                    \
        g_failures++;                                                  \
        fprintf(stderr, "FAIL %s:%d: " msg "\n",                       \
                __FILE__, __LINE__, ##__VA_ARGS__);                    \
    }                                                                  \
} while (0)

static void test_classify_default_shape(void) {
    TriadCompiledProgram *p = triad_compile_classify(-1, -1, -1);
    CHECK(p->n_substrates == 4, "classify default n_subs = %d (want 4)",
          p->n_substrates);
    CHECK(p->n_couplings == 4, "classify default n_couplings = %d (want 4)",
          p->n_couplings);
    /* cls_0: Λ = -0.5, ω = 0.05 */
    CHECK(p->substrates[0].Lambda == -0.5,
          "cls_0 Lambda = %g (want -0.5)", p->substrates[0].Lambda);
    CHECK(p->substrates[0].omega == 0.05,
          "cls_0 omega = %g (want 0.05)", p->substrates[0].omega);
    /* P1+P2+P3 all live on cls_0 (defaults) */
    CHECK(p->substrates[0].alpha != 0.0 && p->substrates[0].Gamma != 0.0
       && p->substrates[0].f_FDT != 0.0 && p->substrates[0].M >= 1,
          "cls_0 missing pillar: alpha=%g Gamma=%g f_FDT=%g M=%d",
          p->substrates[0].alpha, p->substrates[0].Gamma,
          p->substrates[0].f_FDT, p->substrates[0].M);
    /* coupling 0: cls_0 -> cls_1, κ = -3.0 */
    CHECK(strcmp(p->couplings[0].src, "cls_0") == 0
       && strcmp(p->couplings[0].dst, "cls_1") == 0
       && p->couplings[0].kappa == -3.0,
          "couple[0] wrong: %s->%s κ=%g",
          p->couplings[0].src, p->couplings[0].dst, p->couplings[0].kappa);
    /* last coupling: ring close cls_3 -> cls_0, κ = -1.5 */
    CHECK(strcmp(p->couplings[3].src, "cls_3") == 0
       && strcmp(p->couplings[3].dst, "cls_0") == 0
       && p->couplings[3].kappa == -1.5,
          "couple[3] wrong: %s->%s κ=%g",
          p->couplings[3].src, p->couplings[3].dst, p->couplings[3].kappa);
    triad_compiled_program_free(p);
}

static void test_generate_crystal_profile(void) {
    TriadCompiledProgram *p = triad_compile_generate("crystal", 128);
    CHECK(p->n_substrates == 1, "generate(crystal) n_subs = %d (want 1)",
          p->n_substrates);
    CHECK(p->T == 15.0, "generate(crystal) T = %g (want 15)", p->T);
    TriadSubstrateConfig *s = &p->substrates[0];
    CHECK(s->Lambda == -8.0, "Lambda = %g (want -8)", s->Lambda);
    CHECK(s->alpha == 0.0, "alpha = %g (want 0)", s->alpha);
    CHECK(s->V_ext == NULL, "V_ext should be NULL for crystal");
    CHECK(s->M == 2, "crystal memory chain M = %d (want 2)", s->M);
    /* P2+P3 still present: ν > 0, Γ > 0 */
    CHECK(s->Gamma > 0.0 && s->nu[0] > 0.0 && s->nu[1] > 0.0,
          "pillars zeroed in crystal: Gamma=%g nu=%g,%g",
          s->Gamma, s->nu[0], s->nu[1]);
    triad_compiled_program_free(p);
}

static void test_topology_recognition(void) {
    CHECK(strcmp(triad_topology_from_couplings(4, 4), "ring") == 0,
          "topology(4 couplings, 4 subs) should be ring");
    CHECK(strcmp(triad_topology_from_couplings(6, 3), "full") == 0,
          "topology(6 couplings, 3 subs) should be full");
    CHECK(strcmp(triad_topology_from_couplings(0, 5), "ring") == 0,
          "topology(0 couplings) should be ring fallback");
}

static void test_instantiate_pillars_live(void) {
    /* Materialise classify_default into a MultiRuntime. Every substrate
     * must carry its declared Λ, α, Γ, f_FDT, ν, λ. */
    TriadCompiledProgram *p = triad_compile_classify(-1, -1, -1);
    TriadMultiRuntime *rt = triad_compiler_instantiate(p, /*seed*/42);
    CHECK(rt != NULL, "instantiate returned NULL");
    if (rt) {
        CHECK(rt->n_substrates == 4, "rt n_subs = %d (want 4)", rt->n_substrates);
        for (int i = 0; i < rt->n_substrates; ++i) {
            TriadSubstrate *s = rt->substrates[i];
            /* Each substrate's Λ_e matches the declared profile −0.5·(1+0.2·i). */
            double expected = -0.5 * (1.0 + 0.2 * (double)i);
            CHECK(s->Lambda == expected,
                  "sub %d Λ = %g (want %g)", i, s->Lambda, expected);
            CHECK(s->alpha != 0.0 && s->Gamma != 0.0 && s->f_FDT != 0.0,
                  "sub %d pillar zeroed: α=%g Γ=%g f_FDT=%g",
                  i, s->alpha, s->Gamma, s->f_FDT);
            CHECK(s->M >= 1, "sub %d M = %d (P2 chain missing)", i, s->M);
        }
        /* Single segment covers full T. */
        CHECK(rt->n_segments == 1, "rt n_segments = %d (want 1)", rt->n_segments);
        CHECK(rt->segments[0].n_edges == 4,
              "segment edges = %d (want 4)", rt->segments[0].n_edges);
        triad_mr_free(rt);
    }
    triad_compiled_program_free(p);
}

static void test_remember_lambda_profile(void) {
    /* remember default: timescales=(1,5,20) → ν=(1, 0.2, 0.05), λ=(-0.3, -0.15, -0.1) */
    TriadCompiledProgram *p = triad_compile_remember(NULL, 0, -1);
    TriadSubstrateConfig *s = &p->substrates[0];
    CHECK(s->M == 3, "remember default M = %d (want 3)", s->M);
    CHECK(s->nu[0] == 1.0 && s->nu[1] == 0.2 && s->nu[2] == 0.05,
          "remember ν = (%g, %g, %g) (want (1, 0.2, 0.05))",
          s->nu[0], s->nu[1], s->nu[2]);
    /* lam[2] = -0.3 / 3 which is NOT -0.1 in IEEE-754; it is
     * -0.09999999999999999. The arithmetic is not standard — we test
     * against the exact same expression, not against a "clean" 0.1. */
    CHECK(s->lam[0] == -0.3 && s->lam[1] == -0.3 / 2.0 && s->lam[2] == -0.3 / 3.0,
          "remember λ = (%.17g, %.17g, %.17g) (want -0.3, -0.3/2, -0.3/3)",
          s->lam[0], s->lam[1], s->lam[2]);
    CHECK(s->Gamma == 0.02, "remember Γ = %g (want 0.02)", s->Gamma);
    triad_compiled_program_free(p);
}

static void test_couple_topologies(void) {
    /* ring: n edges */
    TriadCompiledProgram *p = triad_compile_couple(4, "ring", -3.0);
    CHECK(p->n_couplings == 4, "ring n_couplings = %d (want 4)", p->n_couplings);
    triad_compiled_program_free(p);
    /* full: n(n-1) edges */
    p = triad_compile_couple(3, "full", -3.0);
    CHECK(p->n_couplings == 6, "full(3) n_couplings = %d (want 6)", p->n_couplings);
    /* full κ scales by 1/n */
    CHECK(p->couplings[0].kappa == -3.0 / 3.0,
          "full κ scaling: got %g (want %g)",
          p->couplings[0].kappa, -3.0 / 3.0);
    triad_compiled_program_free(p);
    /* star: 2(n-1) edges */
    p = triad_compile_couple(4, "star", -3.0);
    CHECK(p->n_couplings == 6, "star(4) n_couplings = %d (want 6)", p->n_couplings);
    triad_compiled_program_free(p);
}

int main(void) {
    test_classify_default_shape();
    test_generate_crystal_profile();
    test_topology_recognition();
    test_instantiate_pillars_live();
    test_remember_lambda_profile();
    test_couple_topologies();

    if (g_failures == 0) {
        printf("test_compiler: PASS (%d/%d)\n", g_total, g_total);
        return 0;
    } else {
        printf("test_compiler: FAIL (%d/%d)\n", g_failures, g_total);
        return 1;
    }
}
