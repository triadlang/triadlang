#ifndef TRIAD_EQUILIBRIUM_H
#define TRIAD_EQUILIBRIUM_H

#include "triad_kernel.h"

typedef enum {
    EQ_HW_CPU     = 0,
    EQ_HW_MEM     = 1,
    EQ_HW_NET     = 2,
    EQ_HW_GPU     = 3,
    EQ_HW_DISK    = 4,
    EQ_HW_THERMAL = 5,
} EqHwDomain;

typedef struct {
    EqHwDomain domain;
    double      utilization;
    double      temperature;
    double      bandwidth;
    double      capacity;
    double      dissipation;
} EqHwState;

typedef struct {
    EqHwState states[6];
    double    total_energy;
    double    fdt_precision;
    double    coupling_strength;
    uint64_t  last_update_tick;
} EqState;

void triad_eq_init(void);
void triad_eq_update(uint64_t tick);
void triad_eq_balance(EqHwDomain a, EqHwDomain b, double *out);
void triad_eq_read(EqHwDomain d, double *out);
void triad_eq_set_coupling(double kappa);
EqState *triad_eq_state(void);

#endif