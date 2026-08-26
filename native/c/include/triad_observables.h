#ifndef TRIAD_OBSERVABLES_H
#define TRIAD_OBSERVABLES_H

#include "triad_rt.h"

double triad_obs_norm(const TriadCplx *psi, int64_t n, double dx);

double triad_obs_peak_density(const TriadCplx *psi, int64_t n);

double triad_obs_fwhm(const TriadCplx *psi, int64_t n, double dx);

double triad_obs_ipr(const TriadCplx *psi, int64_t n, double dx);

double triad_obs_participation_ratio(const TriadCplx *psi, int64_t n, double dx);

void triad_obs_power_spectrum(const TriadCplx *psi, int32_t N, double dx,
                              double *k_out, double *P_out);

double triad_obs_dominant_wavenumber(const TriadCplx *psi, int32_t N, double dx,
                                     double k_min);

double triad_obs_crystallinity(const TriadCplx *psi, int32_t N, double dx,
                                double k_cutoff);

double triad_obs_stabilization_score(const double *arr, int64_t n);

double triad_obs_time_to_stabilize(const double *arr, const double *t_arr,
                                    int64_t n, double tolerance);

#endif

double triad_obs_energy(const TriadCplx *psi, int32_t N, double dx,
                        double hbar, double m, double Lambda,
                        const double *V_ext, const double *V_mem);

double triad_obs_fdt_precision(double f_FDT, double dx, int D);
