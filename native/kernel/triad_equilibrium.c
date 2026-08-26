#include "triad_kernel.h"
#include "triad_equilibrium.h"
#include "triad_kernel_rt.h"

static EqState eq_state;

void triad_eq_init(void) {
    for (int i = 0; i < 6; i++) {
        eq_state.states[i].domain = (EqHwDomain)i;
        eq_state.states[i].utilization = 0.0;
        eq_state.states[i].temperature = 300.0;
        eq_state.states[i].bandwidth = 0.0;
        eq_state.states[i].capacity = 100.0;
        eq_state.states[i].dissipation = 0.0;
    }
    eq_state.total_energy = 0.0;
    eq_state.fdt_precision = 1.0;
    eq_state.coupling_strength = -3.0;
    eq_state.last_update_tick = 0;
}

void triad_eq_update(uint64_t tick) {
    eq_state.last_update_tick = tick;
    double dt = 0.01;
    for (int i = 0; i < 6; i++) {
        EqHwState *s = &eq_state.states[i];
        double noise = 0.002 * ((double)((tick * 1103515245 + i * 12345) & 0xFFFF) / 65535.0 - 0.5);
        double decay = triad_exp(-s->utilization * dt);
        s->dissipation = s->capacity * s->utilization * s->utilization * dt;
        s->utilization = s->utilization * decay + noise;
        if (s->utilization < 0) s->utilization = 0;
        if (s->utilization > 1) s->utilization = 1;
        s->temperature = 300.0 + s->utilization * 50.0 + noise * 10.0;
        eq_state.total_energy += s->dissipation;
    }
    double sum = 0;
    for (int i = 0; i < 6; i++) sum += eq_state.states[i].utilization;
    double mean = sum / 6.0;
    double variance = 0;
    for (int i = 0; i < 6; i++) {
        double d = eq_state.states[i].utilization - mean;
        variance += d * d;
    }
    variance /= 6.0;
    eq_state.fdt_precision = 1.0 / (1.0 + variance * 100.0);
}

void triad_eq_balance(EqHwDomain a, EqHwDomain b, double *out) {
    if (a < 0 || a > 5 || b < 0 || b > 5) { *out = 0.0; return; }
    double ua = eq_state.states[a].utilization;
    double ub = eq_state.states[b].utilization;
    double kappa = eq_state.coupling_strength;
    double du = kappa * (ub - ua) * 0.1;
    eq_state.states[a].utilization += du;
    if (eq_state.states[a].utilization < 0) eq_state.states[a].utilization = 0;
    if (eq_state.states[a].utilization > 1) eq_state.states[a].utilization = 1;
    *out = du;
}

void triad_eq_read(EqHwDomain d, double *out) {
    if (d < 0 || d > 5) { *out = 0.0; return; }
    *out = eq_state.states[d].utilization;
}

void triad_eq_set_coupling(double kappa) {
    eq_state.coupling_strength = kappa;
}

EqState *triad_eq_state(void) {
    return &eq_state;
}