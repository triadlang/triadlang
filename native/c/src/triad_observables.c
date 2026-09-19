#include "triad_observables.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static double _cabs2(TriadCplx z) { return z.re*z.re + z.im*z.im; }

double triad_obs_norm(const TriadCplx *psi, int64_t n, double dx) {
    double s = 0;
    for (int64_t i = 0; i < n; i++) s += _cabs2(psi[i]);
    return s * dx;
}

double triad_obs_peak_density(const TriadCplx *psi, int64_t n) {
    double mx = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = _cabs2(psi[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

double triad_obs_fwhm(const TriadCplx *psi, int64_t n, double dx) {
    double peak = 0;
    int64_t idx_peak = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = _cabs2(psi[i]);
        if (d > peak) { peak = d; idx_peak = i; }
    }
    if (peak <= 0) return 0.0;
    double half = 0.5 * peak;

    int64_t left = idx_peak;
    while (left > 0 && _cabs2(psi[left]) > half) left--;

    int64_t right = idx_peak;
    while (right < n - 1 && _cabs2(psi[right]) > half) right++;

    return (double)(right - left) * dx;
}

double triad_obs_ipr(const TriadCplx *psi, int64_t n, double dx) {
    double n2 = 0, n4 = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = _cabs2(psi[i]);
        n2 += d;
        n4 += d * d;
    }
    n2 *= dx;
    n4 *= dx;
    double denom = n2 * n2;
    return (denom > 1e-30) ? n4 / denom : 0.0;
}

double triad_obs_participation_ratio(const TriadCplx *psi, int64_t n, double dx) {
    double n2 = 0, n4 = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = _cabs2(psi[i]);
        n2 += d;
        n4 += d * d;
    }
    n2 *= dx;
    n4 *= dx;
    return (n4 > 1e-30) ? (n2 * n2) / n4 : 0.0;
}

void triad_obs_power_spectrum(const TriadCplx *psi, int32_t N, double dx,
                              double *k_out, double *P_out) {

    TriadCplx *psi_hat = malloc(sizeof(TriadCplx) * N);
    triad_fft_fn(N, psi, psi_hat);

    double *k_raw = malloc(sizeof(double) * N);
    triad_fftfreq(N, dx, k_raw);

    double *P_raw = malloc(sizeof(double) * N);
    for (int32_t i = 0; i < N; i++)
        P_raw[i] = _cabs2(psi_hat[i]);

    int32_t *order = malloc(sizeof(int32_t) * N);
    for (int32_t i = 0; i < N; i++) order[i] = i;

    for (int32_t i = 1; i < N; i++) {
        int32_t key = order[i];
        double kv = k_raw[key];
        int32_t j = i - 1;
        while (j >= 0 && k_raw[order[j]] > kv) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    for (int32_t i = 0; i < N; i++) {
        k_out[i] = k_raw[order[i]];
        P_out[i] = P_raw[order[i]];
    }

    free(psi_hat);
    free(k_raw);
    free(P_raw);
    free(order);
}

double triad_obs_dominant_wavenumber(const TriadCplx *psi, int32_t N, double dx,
                                     double k_min) {
    double *k = malloc(sizeof(double) * N);
    double *P = malloc(sizeof(double) * N);
    triad_obs_power_spectrum(psi, N, dx, k, P);

    double best_P = -1;
    double best_k = 0;
    for (int32_t i = 0; i < N; i++) {
        if (fabs(k[i]) >= k_min && P[i] > best_P) {
            best_P = P[i];
            best_k = fabs(k[i]);
        }
    }

    free(k);
    free(P);
    return best_k;
}

double triad_obs_crystallinity(const TriadCplx *psi, int32_t N, double dx,
                                double k_cutoff) {
    double *k = malloc(sizeof(double) * N);
    double *P = malloc(sizeof(double) * N);
    triad_obs_power_spectrum(psi, N, dx, k, P);

    double total = 0, structured = 0;
    for (int32_t i = 0; i < N; i++) {
        total += P[i];
        if (fabs(k[i]) > k_cutoff) structured += P[i];
    }

    free(k);
    free(P);
    return (total > 0) ? structured / total : 0.0;
}

static double _cv(const double *arr, int64_t n) {
    if (n < 2) return 0.0;
    double mean = 0;
    for (int64_t i = 0; i < n; i++) mean += arr[i];
    mean /= n;
    if (fabs(mean) < 1e-30) return 0.0;
    double var = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = arr[i] - mean;
        var += d * d;
    }
    var /= n;
    return sqrt(var) / fabs(mean);
}

double triad_obs_stabilization_score(const double *arr, int64_t n) {
    if (n < 4) return 0.0;
    int64_t half = n / 2;
    double cv_e = _cv(arr, half);
    double cv_l = _cv(arr + half, n - half);
    if (cv_e < 1e-12)
        return (cv_l < 1e-12) ? 1.0 : 0.0;
    double score = 1.0 - cv_l / cv_e;
    if (score < 0.0) score = 0.0;
    if (score > 1.0) score = 1.0;
    return score;
}

double triad_obs_time_to_stabilize(const double *arr, const double *t_arr,
                                    int64_t n, double tolerance) {
    if (n == 0) return 0.0;
    int64_t late_start = n * 3 / 4;
    if (late_start < 1) late_start = 1;
    double mean = 0;
    int64_t late_n = n - late_start;
    for (int64_t i = late_start; i < n; i++) mean += arr[i];
    mean /= late_n;

    double band = tolerance * fabs(mean);
    if (fabs(mean) < 1e-30) band = tolerance;

    for (int64_t i = 0; i < n; i++) {
        if (fabs(arr[i] - mean) <= band) {

            int settled = 1;
            for (int64_t j = i; j < n; j++) {
                if (fabs(arr[j] - mean) > band) { settled = 0; break; }
            }
            if (settled) return t_arr[i];
        }
    }
    return t_arr[n - 1];
}

double triad_obs_energy(const TriadCplx *psi, int32_t N, double dx,
                        double hbar, double m, double Lambda,
                        const double *V_ext, const double *V_mem) {

    TriadCplx *psi_k = malloc(sizeof(TriadCplx) * (size_t)N);
    double *kvec = malloc(sizeof(double) * (size_t)N);
    if (!psi_k || !kvec) { free(psi_k); free(kvec); return 0.0; }
    triad_fft_fn(N, psi, psi_k);

    triad_fftfreq(N, dx, kvec);


    double kinetic = 0.0;
    double hbar2_over_2m = hbar * hbar / (2.0 * m);
    for (int32_t i = 0; i < N; i++) {
        double k2 = kvec[i] * kvec[i];
        kinetic += hbar2_over_2m * k2 * _cabs2(psi_k[i]);
    }
    kinetic *= dx / (double)N;


    double nonlin = 0.0;
    for (int32_t i = 0; i < N; i++) {
        double rho = _cabs2(psi[i]);
        nonlin += rho * rho;
    }
    nonlin *= 0.5 * Lambda * dx;


    double pot = 0.0;
    if (V_ext != NULL) {
        for (int32_t i = 0; i < N; i++)
            pot += V_ext[i] * _cabs2(psi[i]);
    }
    if (V_mem != NULL) {
        for (int32_t i = 0; i < N; i++)
            pot += V_mem[i] * _cabs2(psi[i]);
    }
    pot *= dx;

    free(psi_k);
    free(kvec);
    return kinetic + nonlin + pot;
}

double triad_obs_fdt_precision(double f_FDT, double dx, int D) {
    if (f_FDT <= 0.0) return 1e12;
    double dxD = dx;
    if (D == 2) dxD = dx * dx;
    else if (D == 3) dxD = dx * dx * dx;
    double variance = f_FDT / dxD;
    if (variance <= 0.0) return 1e12;
    return 1.0 / variance;
}
