#ifndef _POTENTIAL_H
#define _POTENTIAL_H 1

void potential_sin_6pt(gsl_vector *V, double Vpts[6]);

void potential_test_stationary(const params *params,
			       const gsl_vector *V,
			       const double mass,
			       unsigned int nstates,
			       gsl_vector_complex ***psis,
			       double **Es);

#endif /* defined(_POTENTIAL_H) */

