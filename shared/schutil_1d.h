#ifndef _SCHUTIL_H
#define _SCHUTIL_H 1

void set_hamiltonian(gsl_matrix *H, const gsl_vector *V, const double planck, const double mass, const double hstep);

void set_hamiltonian_circular(gsl_matrix *H, const gsl_vector *V, const double planck, const double mass, const double hstep);

void set_hamiltonian_sq2d(gsl_matrix *H, const gsl_vector *V, const double planck, const double mass, const double hstep, const grid2d *grid);

double get_energy(const gsl_matrix *H, const double hstep, const gsl_vector_complex *psi);

void eigen_solve_alloc(const gsl_matrix *H, gsl_vector **eval, gsl_matrix **evec);
void eigen_norm_state_alloc(const gsl_matrix *evec, const double hstep, int state, gsl_vector_complex **psi_state);

void check_symmetry(const gsl_matrix *M);
void check_unitarity(const gsl_matrix_complex *M, FILE *fdebug);

#endif
