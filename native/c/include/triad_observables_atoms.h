#ifndef TRIAD_OBSERVABLES_ATOMS_H
#define TRIAD_OBSERVABLES_ATOMS_H

#include "triad_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TRIAD_DEFAULT_ATOM_THRESHOLD 0.25

int triad_atom_count_nd(const TriadCplx *psi, int D, int N, double dx,
                        double threshold_frac);

double *triad_atom_centroids_nd(const TriadCplx *psi, int D, int N, double dx,
                                double threshold_frac, int *out_count);

int     triad_atom_count_1d(const TriadCplx *psi, int N, double dx,
                            double threshold_frac);
double *triad_atom_centroids_1d(const TriadCplx *psi, int N, double dx,
                                double threshold_frac, int *out_count);

double  triad_atom_separation_1d(const TriadCplx *psi, int N, double dx,
                                 double threshold_frac);

int     triad_atom_count_2d(const TriadCplx *psi, int N, double dx,
                            double threshold_frac);
int     triad_atom_count_3d(const TriadCplx *psi, int N, double dx,
                            double threshold_frac);
double *triad_atom_centroids_2d(const TriadCplx *psi, int N, double dx,
                                double threshold_frac, int *out_count);
double *triad_atom_centroids_3d(const TriadCplx *psi, int N, double dx,
                                double threshold_frac, int *out_count);

#define triad_count_clusters_1d   triad_atom_count_1d
#define triad_cluster_centroids_1d triad_atom_centroids_1d

double triad_atoms_per_region(const TriadCplx *psi, int D, int N, double dx,
                              double threshold_frac);

double triad_atomicity_ratio(const TriadCplx *psi_macro,
                             const TriadCplx *psi_aggregate,
                             int D, int N, double dx,
                             double threshold_frac);

double triad_atom_persistence_late(const double *density_traj,
                                   int n_records, int N, double dx,
                                   double threshold_frac);

#ifdef __cplusplus
}
#endif

#endif
