#ifndef _SCHUTIL_1D_H
#define _SCHUTIL_1D_H 1

void set_hamiltonian_bounded(gsl_matrix *H, 
			     const params *params,
			     const gsl_vector *V,
			     const double mass);

void set_hamiltonian_circular(gsl_matrix *H, 
			      const params *params,
			      const gsl_vector *V,
			      const double mass);

#endif /* defined(_SCHUTIL_1D) */
