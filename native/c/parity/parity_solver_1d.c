/*
 * parity_solver_1d.c — Native solver_1d fingerprint, byte-identical
 * format to scripts/solver_1d_to_text.py.
 *
 * The §7.4 acceptance criteria call for:
 *   - linear mode: max ||Ψ_A|² - |Ψ_B|²|  ≤ 1e-13 (fp64 floor)
 *   - full, noiseless: relative L2 in |Ψ|² ≤ 1e-3
 * These tolerances are NOT imposed by us — they are the precision floor
 * that two independent IEEE-754 implementations of the same equation
 * naturally reach. The comparator enforces them at the end.
 *
 * Field layout per line matches the Python reference byte-for-byte;
 * floats use triad_py_repr_float (shortest-roundtrip).
 */
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
        if (i + 1 < n) putchar(' ');
    }
    putchar(')');
}

static void build_initial_psi(int N, double L, TriadCplx *psi) {
    double dx = L / N;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double x = -0.5 * L + (double)i * dx;
        double g = exp(-(x * x) / 8.0);
        psi[i].re = g; psi[i].im = 0.0;
        sum += g * g;
    }
    double inv = 1.0 / sqrt(sum * dx);
    for (int i = 0; i < N; ++i) {
        psi[i].re *= inv;
    }
}

static double _norm(const TriadCplx *psi, int N, double dx) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) s += psi[i].re * psi[i].re + psi[i].im * psi[i].im;
    return s * dx;
}

static void dump_run(const char *name, const TriadSolverC *p) {
    int N = p->N;
    double dx = p->L / (double)N;
    TriadCplx *psi0 = (TriadCplx *)malloc(sizeof(TriadCplx) * (size_t)N);
    build_initial_psi(N, p->L, psi0);
    double norm0 = _norm(psi0, N, dx);

    TriadSolverResult r = triad_solve_from_psi(p, psi0);
    free(psi0);

    /* recompute final norm from psi_final for an apples-to-apples
       comparison with Python's recomputation. */
    double normT = _norm(r.psi_final, N, dx);

    printf("fixture %s\n", name);
    char lb[64], db[64], tb[64];
    _fmt(p->L, lb, sizeof lb);
    _fmt(p->dt, db, sizeof db);
    _fmt(p->T, tb, sizeof tb);
    const char *mode_str = "full";
    if (p->mode == 0) mode_str = "linear";
    else if (p->mode == 1) mode_str = "thermal";
    printf("N %d L %s dt %s T %s mode %s\n", N, lb, db, tb, mode_str);

    char ab[64], cb[64], sb[64], gb[64], fb[64];
    _fmt(p->Lambda, ab, sizeof ab);
    _fmt(p->alpha, cb, sizeof cb);
    _fmt(p->sigma, sb, sizeof sb);
    _fmt(p->Gamma, gb, sizeof gb);
    _fmt(p->f_FDT, fb, sizeof fb);
    printf("Lambda %s alpha %s sigma %s Gamma %s f_FDT %s\n",
           ab, cb, sb, gb, fb);

    printf("nu "); _emit_tuple(p->nu, p->M);
    printf(" lam "); _emit_tuple(p->lam, p->M);
    putchar('\n');

    const char *vstr = p->V_ext ? p->V_ext : "None";
    char wb[64]; _fmt(p->omega, wb, sizeof wb);
    printf("V_ext %s omega %s\n", vstr, wb);

    char nb0[64], nbT[64];
    _fmt(norm0, nb0, sizeof nb0);
    _fmt(normT, nbT, sizeof nbT);
    printf("norm_initial %s norm_final %s\n", nb0, nbT);

    /* density on the grid (length N). */
    printf("density");
    char vb[64];
    for (int i = 0; i < N; ++i) {
        double rho = r.psi_final[i].re * r.psi_final[i].re
                   + r.psi_final[i].im * r.psi_final[i].im;
        _fmt(rho, vb, sizeof vb);
        printf(" %s", vb);
    }
    putchar('\n');

    triad_solver_result_free(&r);
}

int main(void) {
    /* Fixture 1 — linear. mode=0 zeroes Λ, α, Γ, f_FDT inside the
     * solver via _effective_params; we still pass the structural values
     * here so the printed parameters match Python's labels. */
    TriadSolverC p_lin = {
        .N = 64, .L = 32.0, .dt = 0.005, .T = 0.5,
        .hbar = 1.0, .m = 1.0, .omega = 0.0,
        .Lambda = 0.0, .alpha = 0.0, .sigma = 2.0,
        .Gamma = 0.0, .f_FDT = 0.0,
        .M = 0, .nu = NULL, .lam = NULL,
        .mode = 0,  /* linear */
        .seed = 42,
        .V_ext = NULL,
        .D = 1,
        .bc = NULL, .bc_width = 0.0,
    };
    dump_run("linear", &p_lin);

    /* Fixture 2 — full noiseless with harmonic V_ext (the only named
     * 1D template the canonical Python solver supports). */
    double nu_full[3]  = { 2.0, 0.5, 0.1 };
    double lam_full[3] = { -0.3, -0.2, -0.1 };
    TriadSolverC p_full = {
        .N = 64, .L = 32.0, .dt = 0.005, .T = 0.5,
        .hbar = 1.0, .m = 1.0, .omega = 0.05,
        .Lambda = -0.5, .alpha = 0.15, .sigma = 1.5,
        .Gamma = 0.05, .f_FDT = 0.0,
        .M = 3, .nu = nu_full, .lam = lam_full,
        .mode = 2,  /* full */
        .seed = 42,
        .V_ext = "harmonic",
        .D = 1,
        .bc = NULL, .bc_width = 0.0,
    };
    dump_run("full_noiseless", &p_full);

    return 0;
}
