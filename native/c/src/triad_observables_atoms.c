#define _POSIX_C_SOURCE 200809L
#include "triad_observables_atoms.h"
#include "triad_rt.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static double _max_rho(const TriadCplx *psi, int64_t n) {
    double m = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double r = psi[i].re * psi[i].re + psi[i].im * psi[i].im;
        if (r > m) m = r;
    }
    return m;
}

int triad_atom_count_1d(const TriadCplx *psi, int N, double dx,
                        double threshold_frac) {
    (void)dx;
    if (N <= 0) return 0;
    double rho_max = _max_rho(psi, N);
    if (rho_max <= 0.0) return 0;
    double thr = threshold_frac * rho_max;
    int *above = (int *)malloc(sizeof(int) * (size_t)N);
    int any = 0, all = 1;
    for (int i = 0; i < N; ++i) {
        double r = psi[i].re * psi[i].re + psi[i].im * psi[i].im;
        above[i] = (r > thr) ? 1 : 0;
        if (above[i]) any = 1;
        else all = 0;
    }
    if (!any) { free(above); return 0; }
    if (all)  { free(above); return 1; }

    int idx0 = 0;
    for (int i = 0; i < N; ++i) {
        if (!above[i]) { idx0 = i; break; }
    }

    int count = 0;
    for (int k = 0; k < N - 1; ++k) {
        int a = above[(idx0 + k) % N];
        int b = above[(idx0 + k + 1) % N];
        if (b - a == 1) count++;
    }
    free(above);
    return count;
}

double *triad_atom_centroids_1d(const TriadCplx *psi, int N, double dx,
                                double threshold_frac, int *out_count) {
    if (out_count) *out_count = 0;
    if (N <= 0) return NULL;
    double rho_max = _max_rho(psi, N);
    if (rho_max <= 0.0) return NULL;
    double thr = threshold_frac * rho_max;

    int n_clusters = 0;
    int in_cluster = 0;
    for (int i = 0; i < N; ++i) {
        double r = psi[i].re * psi[i].re + psi[i].im * psi[i].im;
        int above = (r > thr) ? 1 : 0;
        if (above) {
            if (!in_cluster) { n_clusters++; in_cluster = 1; }
        } else {
            in_cluster = 0;
        }
    }
    if (n_clusters == 0) return NULL;
    double *out = (double *)malloc(sizeof(double) * (size_t)n_clusters);
    int idx = 0;
    in_cluster = 0;
    double sum_xw = 0.0, sum_w = 0.0;
    for (int i = 0; i < N; ++i) {
        double r = psi[i].re * psi[i].re + psi[i].im * psi[i].im;
        int above = (r > thr) ? 1 : 0;
        double x = ((double)i - (double)(N / 2)) * dx;
        if (above) {
            if (!in_cluster) { sum_xw = 0.0; sum_w = 0.0; in_cluster = 1; }
            sum_xw += x * r;
            sum_w += r;
        } else {
            if (in_cluster) {
                double denom = (sum_w > 1e-30) ? sum_w : 1e-30;
                out[idx++] = sum_xw / denom;
                in_cluster = 0;
            }
        }
    }
    if (in_cluster) {
        double denom = (sum_w > 1e-30) ? sum_w : 1e-30;
        out[idx++] = sum_xw / denom;
    }
    *out_count = idx;
    return out;
}

double triad_atom_separation_1d(const TriadCplx *psi, int N, double dx,
                                double threshold_frac) {
    int n;
    double *cs = triad_atom_centroids_1d(psi, N, dx, threshold_frac, &n);
    if (n < 2) { free(cs); return 0.0; }

    for (int i = 1; i < n; ++i) {
        double v = cs[i];
        int j = i - 1;
        while (j >= 0 && cs[j] > v) { cs[j+1] = cs[j]; j--; }
        cs[j+1] = v;
    }
    double mean = 0.0;
    for (int i = 1; i < n; ++i) mean += cs[i] - cs[i-1];
    mean /= (double)(n - 1);
    free(cs);
    return mean;
}

typedef struct { int *parent; int n; } UF;

static void uf_init(UF *u, int n) {
    u->n = n;
    u->parent = (int *)malloc(sizeof(int) * (size_t)(n + 1));
    for (int i = 0; i <= n; ++i) u->parent[i] = i;
}

static int uf_find(UF *u, int x) {
    while (u->parent[x] != x) {
        u->parent[x] = u->parent[u->parent[x]];
        x = u->parent[x];
    }
    return x;
}

static void uf_union(UF *u, int a, int b) {
    int ra = uf_find(u, a), rb = uf_find(u, b);
    if (ra != rb) u->parent[ra] = rb;
}

static void uf_free(UF *u) { free(u->parent); }

static int *_label_2d_nonperiodic(const int *above, int N, int *out_count) {
    int *labels = (int *)calloc((size_t)N * (size_t)N, sizeof(int));
    int *queue  = (int *)malloc(sizeof(int) * (size_t)N * (size_t)N);
    int next_label = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            if (!above[idx] || labels[idx] != 0) continue;
            ++next_label;
            int head = 0, tail = 0;
            queue[tail++] = idx;
            labels[idx] = next_label;
            while (head < tail) {
                int p = queue[head++];
                int pi = p / N, pj = p % N;
                static const int offs[4][2] = { {1,0},{-1,0},{0,1},{0,-1} };
                for (int d = 0; d < 4; ++d) {
                    int ni = pi + offs[d][0];
                    int nj = pj + offs[d][1];
                    if (ni < 0 || ni >= N || nj < 0 || nj >= N) continue;
                    int nidx = ni * N + nj;
                    if (above[nidx] && labels[nidx] == 0) {
                        labels[nidx] = next_label;
                        queue[tail++] = nidx;
                    }
                }
            }
        }
    }
    free(queue);
    *out_count = next_label;
    return labels;
}

static int *_periodic_merge_2d(int *labels, int N, int n, int *final_n) {
    if (n <= 1) { *final_n = n; return labels; }
    UF u; uf_init(&u, n);

    for (int j = 0; j < N; ++j) {
        int lo = labels[0 * N + j];
        int hi = labels[(N - 1) * N + j];
        if (lo > 0 && hi > 0) uf_union(&u, lo, hi);
    }

    for (int i = 0; i < N; ++i) {
        int lo = labels[i * N + 0];
        int hi = labels[i * N + (N - 1)];
        if (lo > 0 && hi > 0) uf_union(&u, lo, hi);
    }

    int *remap = (int *)calloc((size_t)(n + 1), sizeof(int));
    int next_id = 0;
    int *new_labels = (int *)calloc((size_t)N * (size_t)N, sizeof(int));
    for (int i = 0; i < N * N; ++i) {
        int l = labels[i];
        if (l == 0) continue;
        int root = uf_find(&u, l);
        if (remap[root] == 0) remap[root] = ++next_id;
        new_labels[i] = remap[root];
    }
    free(remap);
    uf_free(&u);
    free(labels);
    *final_n = next_id;
    return new_labels;
}

int triad_atom_count_2d(const TriadCplx *psi, int N, double dx,
                        double threshold_frac) {
    (void)dx;
    if (N <= 0) return 0;
    int64_t G = (int64_t)N * N;
    double rho_max = _max_rho(psi, G);
    if (rho_max <= 0.0) return 0;
    double thr = threshold_frac * rho_max;
    int *above = (int *)malloc(sizeof(int) * (size_t)G);
    int any = 0;
    for (int64_t i = 0; i < G; ++i) {
        double r = psi[i].re * psi[i].re + psi[i].im * psi[i].im;
        above[i] = (r > thr) ? 1 : 0;
        if (above[i]) any = 1;
    }
    if (!any) { free(above); return 0; }
    int n;
    int *labels = _label_2d_nonperiodic(above, N, &n);
    free(above);
    int final_n;
    int *merged = _periodic_merge_2d(labels, N, n, &final_n);
    free(merged);
    return final_n;
}

double *triad_atom_centroids_2d(const TriadCplx *psi, int N, double dx,
                                double threshold_frac, int *out_count) {
    if (out_count) *out_count = 0;
    if (N <= 0) return NULL;
    int64_t G = (int64_t)N * N;
    double rho_max = _max_rho(psi, G);
    if (rho_max <= 0.0) return NULL;
    double thr = threshold_frac * rho_max;
    int *above = (int *)malloc(sizeof(int) * (size_t)G);
    int any = 0;
    double *rho = (double *)malloc(sizeof(double) * (size_t)G);
    for (int64_t i = 0; i < G; ++i) {
        rho[i] = psi[i].re * psi[i].re + psi[i].im * psi[i].im;
        above[i] = (rho[i] > thr) ? 1 : 0;
        if (above[i]) any = 1;
    }
    if (!any) { free(above); free(rho); return NULL; }
    int n;
    int *labels = _label_2d_nonperiodic(above, N, &n);
    free(above);
    int final_n;
    int *merged = _periodic_merge_2d(labels, N, n, &final_n);
    if (final_n == 0) { free(merged); free(rho); return NULL; }

    double *sum_iw = (double *)calloc((size_t)final_n, sizeof(double));
    double *sum_jw = (double *)calloc((size_t)final_n, sizeof(double));
    double *sum_w  = (double *)calloc((size_t)final_n, sizeof(double));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            int l = merged[idx];
            if (l == 0) continue;
            double w = rho[idx];
            sum_iw[l-1] += i * w;
            sum_jw[l-1] += j * w;
            sum_w[l-1]  += w;
        }
    }
    free(rho); free(merged);

    double *out = (double *)malloc(sizeof(double) * (size_t)final_n * 2);
    int half = N / 2;
    for (int k = 0; k < final_n; ++k) {
        double denom = sum_w[k] > 0 ? sum_w[k] : 1.0;
        out[k*2 + 0] = (sum_iw[k] / denom - (double)half) * dx;
        out[k*2 + 1] = (sum_jw[k] / denom - (double)half) * dx;
    }
    free(sum_iw); free(sum_jw); free(sum_w);
    *out_count = final_n;
    return out;
}

static int *_label_3d_nonperiodic(const int *above, int N, int *out_count) {
    int64_t G = (int64_t)N * N * N;
    int *labels = (int *)calloc((size_t)G, sizeof(int));
    int *queue  = (int *)malloc(sizeof(int) * (size_t)G);
    int next_label = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int64_t idx = (int64_t)i*N*N + (int64_t)j*N + k;
                if (!above[idx] || labels[idx] != 0) continue;
                ++next_label;
                int head = 0, tail = 0;
                queue[tail++] = (int)idx;
                labels[idx] = next_label;
                while (head < tail) {
                    int p = queue[head++];
                    int pk = p % N;
                    int pj = (p / N) % N;
                    int pi = p / (N * N);
                    static const int offs[6][3] = {
                        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}
                    };
                    for (int d = 0; d < 6; ++d) {
                        int ni = pi + offs[d][0];
                        int nj = pj + offs[d][1];
                        int nk = pk + offs[d][2];
                        if (ni<0||ni>=N||nj<0||nj>=N||nk<0||nk>=N) continue;
                        int nidx = ni*N*N + nj*N + nk;
                        if (above[nidx] && labels[nidx] == 0) {
                            labels[nidx] = next_label;
                            queue[tail++] = nidx;
                        }
                    }
                }
            }
        }
    }
    free(queue);
    *out_count = next_label;
    return labels;
}

static int *_periodic_merge_3d(int *labels, int N, int n, int *final_n) {
    if (n <= 1) { *final_n = n; return labels; }
    UF u; uf_init(&u, n);

    for (int j = 0; j < N; ++j) for (int k = 0; k < N; ++k) {
        int lo = labels[0*N*N + j*N + k];
        int hi = labels[(N-1)*N*N + j*N + k];
        if (lo > 0 && hi > 0) uf_union(&u, lo, hi);
    }

    for (int i = 0; i < N; ++i) for (int k = 0; k < N; ++k) {
        int lo = labels[i*N*N + 0*N + k];
        int hi = labels[i*N*N + (N-1)*N + k];
        if (lo > 0 && hi > 0) uf_union(&u, lo, hi);
    }

    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
        int lo = labels[i*N*N + j*N + 0];
        int hi = labels[i*N*N + j*N + (N-1)];
        if (lo > 0 && hi > 0) uf_union(&u, lo, hi);
    }
    int *remap = (int *)calloc((size_t)(n + 1), sizeof(int));
    int next_id = 0;
    int64_t G = (int64_t)N * N * N;
    int *new_labels = (int *)calloc((size_t)G, sizeof(int));
    for (int64_t i = 0; i < G; ++i) {
        int l = labels[i];
        if (l == 0) continue;
        int root = uf_find(&u, l);
        if (remap[root] == 0) remap[root] = ++next_id;
        new_labels[i] = remap[root];
    }
    free(remap);
    uf_free(&u);
    free(labels);
    *final_n = next_id;
    return new_labels;
}

int triad_atom_count_3d(const TriadCplx *psi, int N, double dx,
                        double threshold_frac) {
    (void)dx;
    if (N <= 0) return 0;
    int64_t G = (int64_t)N * N * N;
    double rho_max = _max_rho(psi, G);
    if (rho_max <= 0.0) return 0;
    double thr = threshold_frac * rho_max;
    int *above = (int *)malloc(sizeof(int) * (size_t)G);
    int any = 0;
    for (int64_t i = 0; i < G; ++i) {
        double r = psi[i].re * psi[i].re + psi[i].im * psi[i].im;
        above[i] = (r > thr) ? 1 : 0;
        if (above[i]) any = 1;
    }
    if (!any) { free(above); return 0; }
    int n;
    int *labels = _label_3d_nonperiodic(above, N, &n);
    free(above);
    int final_n;
    int *merged = _periodic_merge_3d(labels, N, n, &final_n);
    free(merged);
    return final_n;
}

double *triad_atom_centroids_3d(const TriadCplx *psi, int N, double dx,
                                double threshold_frac, int *out_count) {
    if (out_count) *out_count = 0;
    if (N <= 0) return NULL;
    int64_t G = (int64_t)N * N * N;
    double rho_max = _max_rho(psi, G);
    if (rho_max <= 0.0) return NULL;
    double thr = threshold_frac * rho_max;
    int *above = (int *)malloc(sizeof(int) * (size_t)G);
    double *rho = (double *)malloc(sizeof(double) * (size_t)G);
    int any = 0;
    for (int64_t i = 0; i < G; ++i) {
        rho[i] = psi[i].re * psi[i].re + psi[i].im * psi[i].im;
        above[i] = (rho[i] > thr) ? 1 : 0;
        if (above[i]) any = 1;
    }
    if (!any) { free(above); free(rho); return NULL; }
    int n;
    int *labels = _label_3d_nonperiodic(above, N, &n);
    free(above);
    int final_n;
    int *merged = _periodic_merge_3d(labels, N, n, &final_n);
    if (final_n == 0) { free(merged); free(rho); return NULL; }

    double *sum_iw = (double *)calloc((size_t)final_n, sizeof(double));
    double *sum_jw = (double *)calloc((size_t)final_n, sizeof(double));
    double *sum_kw = (double *)calloc((size_t)final_n, sizeof(double));
    double *sum_w  = (double *)calloc((size_t)final_n, sizeof(double));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int64_t idx = (int64_t)i*N*N + (int64_t)j*N + k;
                int l = merged[idx];
                if (l == 0) continue;
                double w = rho[idx];
                sum_iw[l-1] += i * w;
                sum_jw[l-1] += j * w;
                sum_kw[l-1] += k * w;
                sum_w[l-1]  += w;
            }
        }
    }
    free(rho); free(merged);
    double *out = (double *)malloc(sizeof(double) * (size_t)final_n * 3);
    int half = N / 2;
    for (int kk = 0; kk < final_n; ++kk) {
        double denom = sum_w[kk] > 0 ? sum_w[kk] : 1.0;
        out[kk*3 + 0] = (sum_iw[kk] / denom - (double)half) * dx;
        out[kk*3 + 1] = (sum_jw[kk] / denom - (double)half) * dx;
        out[kk*3 + 2] = (sum_kw[kk] / denom - (double)half) * dx;
    }
    free(sum_iw); free(sum_jw); free(sum_kw); free(sum_w);
    *out_count = final_n;
    return out;
}

int triad_atom_count_nd(const TriadCplx *psi, int D, int N, double dx,
                        double threshold_frac) {
    if (D == 1) return triad_atom_count_1d(psi, N, dx, threshold_frac);
    if (D == 2) return triad_atom_count_2d(psi, N, dx, threshold_frac);
    if (D == 3) return triad_atom_count_3d(psi, N, dx, threshold_frac);
    return 0;
}

double *triad_atom_centroids_nd(const TriadCplx *psi, int D, int N, double dx,
                                double threshold_frac, int *out_count) {
    if (D == 1) return triad_atom_centroids_1d(psi, N, dx, threshold_frac, out_count);
    if (D == 2) return triad_atom_centroids_2d(psi, N, dx, threshold_frac, out_count);
    if (D == 3) return triad_atom_centroids_3d(psi, N, dx, threshold_frac, out_count);
    if (out_count) *out_count = 0;
    return NULL;
}

double triad_atoms_per_region(const TriadCplx *psi, int D, int N, double dx,
                              double threshold_frac) {
    if (N <= 0) return 0.0;
    int64_t G = D == 1 ? N : (D == 2 ? (int64_t)N * N : (int64_t)N * N * N);
    double rho_max = _max_rho(psi, G);
    if (rho_max <= 0.0) return 0.0;
    double thr = threshold_frac * rho_max;
    int n_above = 0;
    for (int64_t i = 0; i < G; ++i) {
        double r = psi[i].re * psi[i].re + psi[i].im * psi[i].im;
        if (r > thr) n_above++;
    }
    double dxD = pow(dx, D);
    double occupied = (double)n_above * dxD;
    if (occupied <= 0.0) return 0.0;
    int n_at = triad_atom_count_nd(psi, D, N, dx, threshold_frac);
    return (double)n_at / occupied;
}

double triad_atomicity_ratio(const TriadCplx *psi_macro,
                             const TriadCplx *psi_aggregate,
                             int D, int N, double dx,
                             double threshold_frac) {
    double m = triad_atoms_per_region(psi_macro, D, N, dx, threshold_frac);
    double a = triad_atoms_per_region(psi_aggregate, D, N, dx, threshold_frac);
    if (a <= 0.0) return 0.0;
    return m / a;
}

double triad_atom_persistence_late(const double *density_traj,
                                   int n_records, int N, double dx,
                                   double threshold_frac) {
    (void)dx;
    if (n_records < 4) return 0.0;
    int late_start = n_records * 3 / 4;
    int n_late = n_records - late_start;
    if (n_late == 0) return 0.0;

    int *counts = (int *)malloc(sizeof(int) * (size_t)n_late);
    int *above = (int *)malloc(sizeof(int) * (size_t)N);
    for (int k = 0; k < n_late; ++k) {
        const double *rho = &density_traj[(int64_t)(late_start + k) * N];
        double rmax = 0.0;
        for (int i = 0; i < N; ++i) if (rho[i] > rmax) rmax = rho[i];
        if (rmax <= 0.0) { counts[k] = 0; continue; }
        double thr = threshold_frac * rmax;
        int any = 0, all = 1;
        for (int i = 0; i < N; ++i) {
            above[i] = (rho[i] > thr) ? 1 : 0;
            if (above[i]) any = 1; else all = 0;
        }
        if (!any) { counts[k] = 0; continue; }
        if (all)  { counts[k] = 1; continue; }
        int idx0 = 0;
        for (int i = 0; i < N; ++i) if (!above[i]) { idx0 = i; break; }
        int c = 0;
        for (int kk = 0; kk < N - 1; ++kk) {
            int a = above[(idx0 + kk) % N];
            int b = above[(idx0 + kk + 1) % N];
            if (b - a == 1) c++;
        }
        counts[k] = c;
    }
    free(above);

    double mean = 0.0;
    for (int k = 0; k < n_late; ++k) mean += counts[k];
    mean /= (double)n_late;
    double var = 0.0;
    for (int k = 0; k < n_late; ++k) {
        double d = (double)counts[k] - mean;
        var += d * d;
    }
    var /= (double)n_late;
    free(counts);
    return var;
}
