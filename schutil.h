#ifndef _SCHUTIL_H
#define _SCHUTIL_H 1

#include "grid2d.h"

#define ASSERT_SQUARE(m, msg) { if ((m->size1) != (m->size2)) { fprintf(stderr, "%s (%lu x %lu)", msg, m->size1, m->size2); exit(1); } }
#define ASSERT_SIZE1(m, sz, msg) { if ((m->size1 != sz)) { fprintf(stderr, "%s (%lu != %lu)", msg, m->size1, sz); exit(1); } }
#define ASSERT_SQUARE_SIZE(m, sz, msg) { if ((m->size1 != m->size2) || (m->size1 != sz)) { fprintf(stderr, "%s (%lu x %lu != %lu)", msg, m->size1, m->size2, sz); exit(1); } }

void write_potential(const char *prefix, const gsl_vector *V);
void write_hamiltonian(const char *prefix, const gsl_matrix *H);
void write_energies(const char *prefix, const gsl_matrix *Psis, const gsl_vector *Es);
void write_psi(const char *prefix, const gsl_matrix *Psis, const gsl_vector *Es);

void set_hamiltonian(gsl_matrix *H, const gsl_vector *V, const double planck, const double mass, const double hstep);

void set_hamiltonian_circular(gsl_matrix *H, const gsl_vector *V, const double planck, const double mass, const double hstep);

void set_hamiltonian_sq2d(gsl_matrix *H, const gsl_vector *V, const double planck, const double mass, const double hstep, const grid2d *grid);

double get_energy(const gsl_matrix *H, const double hstep, const gsl_vector_complex *psi);

/* Time evolution matrices */
/* H0, H1 = Hamiltonians at t0, t1
 * Make H = (H0 + H1) / 2
 * A = ((i / tstep) I - 0.5 H), cf i hbar dPsi/dt - 0.5 Havg
 * B = ((i / tstep) I + 0.5 H), cf i hbar dPsi/dt + 0.5 Havg
 * A Psi(t1) = B Psi(t0) OR Psi(t1) = A^{-1} B Psi(t0)
 * U = A^{-1} B
 */

typedef struct {
  gsl_matrix_complex *A;
  gsl_matrix_complex *B;

  gsl_complex ihdt;
  gsl_matrix_complex *halfH;

  gsl_matrix_complex *ALU; /* LU decomp of A */
  gsl_permutation *ALUp;   
} timeevol_halves;

timeevol_halves *timeevol_halves_alloc(const size_t);
void timeevol_halves_free(timeevol_halves *);

void set_timeevol_halves(timeevol_halves *U,
		     const gsl_matrix *H0, const gsl_matrix *H1,
		     const double planck, const double tstep, FILE *fdebug);

void set_timeevol(gsl_matrix_complex *U, const gsl_matrix *H0, const gsl_matrix *H1,
	        const double planck, const double hstep, const double tstep, FILE *fdebug);

void timeevol_state(gsl_vector_complex *, const timeevol_halves *, const gsl_vector_complex *);

void eigen_solve_alloc(const gsl_matrix *H, gsl_vector **eval, gsl_matrix **evec);
void eigen_norm_state_alloc(const gsl_matrix *evec, const double hstep, int state, gsl_vector_complex **psi_state);

void check_symmetry(const gsl_matrix *M);
void check_unitarity(const gsl_matrix_complex *M, FILE *fdebug);

#endif
