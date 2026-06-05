#ifndef TRIAD_OBSERVABLES_H
#define TRIAD_OBSERVABLES_H

/*
 * TriadLang Native Observables (§8 of the triad reference).
 *
 * All observables read from the field produced by P1+P2+P3.
 * They do not modify the field.
 *
 * Field-level scalars (§8.1):   norm, peak_density, fwhm, ipr, participation_ratio
 * Spectral diagnostics (§8.3):  power_spectrum, dominant_wavenumber, crystallinity
 * Time-aggregated (§8.5):       stabilization_score, time_to_stabilize
 */

#include "triad_rt.h"

/* ── Field-level scalars ── */

/* Integral of |psi|^2 over the domain. */
double triad_obs_norm(const TriadCplx *psi, int64_t n, double dx);

/* Maximum |psi|^2 across the grid. */
double triad_obs_peak_density(const TriadCplx *psi, int64_t n);

/* Full-width-at-half-maximum of |psi|^2. Returns 0 if no clear peak. */
double triad_obs_fwhm(const TriadCplx *psi, int64_t n, double dx);

/* Inverse participation ratio: integral(|psi|^4) / (integral(|psi|^2))^2 */
double triad_obs_ipr(const TriadCplx *psi, int64_t n, double dx);

/* Effective support: 1 / IPR */
double triad_obs_participation_ratio(const TriadCplx *psi, int64_t n, double dx);

/* ── Spectral diagnostics ── */

/* Power spectrum via 1D FFT. Writes k[] and P[] arrays of length N.
 * Caller must allocate k_out and P_out of size N.
 * Returns sorted ascending by k. */
void triad_obs_power_spectrum(const TriadCplx *psi, int32_t N, double dx,
                              double *k_out, double *P_out);

/* k* = location of peak |Psi_hat(k)|^2 for |k| >= k_min. */
double triad_obs_dominant_wavenumber(const TriadCplx *psi, int32_t N, double dx,
                                     double k_min);

/* Fraction of spectral power at |k| > k_cutoff. */
double triad_obs_crystallinity(const TriadCplx *psi, int32_t N, double dx,
                                double k_cutoff);

/* ── Time-aggregated metrics ── */

/* Stabilization score: 1 - cv_late/cv_early, clamped [0,1].
 * arr is an observable time series of length n. */
double triad_obs_stabilization_score(const double *arr, int64_t n);

/* First t at which observable enters and stays within tolerance of late mean.
 * arr and t_arr are both of length n. */
double triad_obs_time_to_stabilize(const double *arr, const double *t_arr,
                                    int64_t n, double tolerance);

#endif /* TRIAD_OBSERVABLES_H */
