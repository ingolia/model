#ifndef _SCHUTIL_H
#define _SCHUTIL_H 1

double get_energy(const gsl_matrix *H, const double hstep, const gsl_vector_complex *psi);

void eigen_solve_alloc(const gsl_matrix *H, gsl_vector **eval, gsl_matrix **evec);
void eigen_norm_state_alloc(const gsl_matrix *evec, const double hstep, int state, gsl_vector_complex **psi_state);

#endif /* defined(_SCHUTIL_H) */
