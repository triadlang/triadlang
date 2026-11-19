#ifndef TRIAD_OBSERVABLES_ATOMS_H
#define TRIAD_OBSERVABLES_ATOMS_H

/* ════════════════════════════════════════════════════════════════════
   TriadLang — Atom / cluster observables (Part II §8 + §Observables)
   Native C port of runtime/observables_atoms.py.

   "An atom is a persistent extended entity": a connected blob of |Ψ|²
   above threshold_frac · max(|Ψ|²). The threshold is taken on the
   density field that comes out of the FULL equation P1+P2+P3 — atoms
   are not imposed by the observable; they are READ from a field whose
   structure emerges from chaos+equilibrium under the equation itself.
   ════════════════════════════════════════════════════════════════════ */

#include "triad_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TRIAD_DEFAULT_ATOM_THRESHOLD 0.25

/* ── ND dispatchers ─────────────────────────────────────────────── */

/* atom_count_nd(psi, dx, threshold_frac): counts connected blobs.
 *   D = 1: 1D ring (periodic boundaries).
 *   D = 2: 2D row-major (i,j) with 4-connectivity, periodic BCs.
 *   D = 3: 3D row-major (i,j,k) with 6-connectivity, periodic BCs.
 * Returns 0 if the field is null or no point exceeds threshold. */
int triad_atom_count_nd(const TriadCplx *psi, int D, int N, double dx,
                        double threshold_frac);

/* atom_centroids_nd(psi, dx, threshold_frac, *out_count):
 *   Writes the centroid coordinates of each atom into a freshly-malloc'd
 *   buffer of size (*out_count) * D doubles. Layout: [a0.x, a0.y, a0.z, a1.x, ...]
 *   (Y/Z absent when D < 3.) Each coordinate is in the world frame with
 *   origin at the centre of the box (matches runtime/observables_atoms.py).
 *   Centroids are sorted lexicographically (x, then y, then z) so the
 *   output order matches Python after sorting on the test side — needed
 *   because the labelling order differs between scipy.ndimage and our
 *   pure-C BFS, but the SET of centroids must be identical.
 *   The caller frees the returned buffer with free(). Returns NULL when
 *   *out_count == 0. */
double *triad_atom_centroids_nd(const TriadCplx *psi, int D, int N, double dx,
                                double threshold_frac, int *out_count);

/* ── 1D specialisations (also dispatched by _nd above) ─────────── */

int     triad_atom_count_1d(const TriadCplx *psi, int N, double dx,
                            double threshold_frac);
double *triad_atom_centroids_1d(const TriadCplx *psi, int N, double dx,
                                double threshold_frac, int *out_count);
/* Mean nearest-neighbour gap between sorted 1D centroids. 0 if < 2 atoms. */
double  triad_atom_separation_1d(const TriadCplx *psi, int N, double dx,
                                 double threshold_frac);

/* ── 2D / 3D ────────────────────────────────────────────────────── */

int     triad_atom_count_2d(const TriadCplx *psi, int N, double dx,
                            double threshold_frac);
int     triad_atom_count_3d(const TriadCplx *psi, int N, double dx,
                            double threshold_frac);
double *triad_atom_centroids_2d(const TriadCplx *psi, int N, double dx,
                                double threshold_frac, int *out_count);
double *triad_atom_centroids_3d(const TriadCplx *psi, int N, double dx,
                                double threshold_frac, int *out_count);

/* ── Aliases (atom_count == count_clusters in the framework) ──── */
#define triad_count_clusters_1d   triad_atom_count_1d
#define triad_cluster_centroids_1d triad_atom_centroids_1d

/* ── Composite observables ─────────────────────────────────────── */

/* atoms_per_region: n_at / occupied_support where occupied_support is
 * the volume (Σ above) * dx^D. Returns 0 if no atoms. */
double triad_atoms_per_region(const TriadCplx *psi, int D, int N, double dx,
                              double threshold_frac);

/* atomicity_ratio: atoms_per_region(macro) / atoms_per_region(aggregate).
 * Returns 0 if aggregate is empty. */
double triad_atomicity_ratio(const TriadCplx *psi_macro,
                             const TriadCplx *psi_aggregate,
                             int D, int N, double dx,
                             double threshold_frac);

/* atom_persistence_late: variance of atom_count over the late-time
 * quarter of density_traj. density_traj is row-major shape
 *   (n_records, grid_size) with density_traj[k * grid_size + i].
 * For 1D grid_size = N. Returns 0 if n_records < 4. */
double triad_atom_persistence_late(const double *density_traj,
                                   int n_records, int N, double dx,
                                   double threshold_frac);

#ifdef __cplusplus
}
#endif

#endif /* TRIAD_OBSERVABLES_ATOMS_H */
