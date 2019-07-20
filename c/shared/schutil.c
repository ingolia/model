#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <unistd.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_complex_double.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_vector.h>

#include "params.h"
#include "schutil.h"

// <psi' | H | psi>
// Take psi = a + bi for a, b real and psi' = a - bi
// <psi' | H | a+bi> = <psi' | H | a> + i <psi' | H | b>
//                   = <a|H|a> - i <b|H|a> + i <a|H|b> + i (-i <b|H|b>)
// H symmetric, <b|H|a> = <a|H|b> and kill imaginary terms
//                   = <a|H|a> + <b|H|b>
double get_energy(const gsl_matrix *H, const params *params, const gsl_vector_complex *psi)
{
  gsl_vector_const_view a = gsl_vector_complex_const_real(psi);
  gsl_vector_const_view b = gsl_vector_complex_const_imag(psi);

  gsl_vector *q = gsl_vector_alloc(psi->size);
  double aHa;
  gsl_blas_dsymv(CblasUpper, 1.0, H, &(a.vector), 0.0, q);
  gsl_blas_ddot(&(a.vector), q, &aHa);

  double bHb;
  gsl_blas_dsymv(CblasUpper, 1.0, H, &(b.vector), 0.0, q);
  gsl_blas_ddot(&(b.vector), q, &bHb);
  
  gsl_vector_free(q);

  return (aHa + bHb) * params->hstep;
}

void eigen_solve_alloc(const gsl_matrix *Hin, gsl_vector **eval, gsl_matrix **evec)
{
  const size_t STATESIZE = Hin->size1;
  if (Hin->size2 != STATESIZE) {
    fprintf(stderr, "eigen_solve_alloc: Hin not square");
    exit(1);
  }

  gsl_matrix *H = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_matrix_memcpy(H, Hin);
  
  *eval = gsl_vector_alloc(STATESIZE);
  *evec = gsl_matrix_alloc(STATESIZE, STATESIZE);

  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(STATESIZE);

  gsl_eigen_symmv(H, *eval, *evec, w);

  gsl_eigen_symmv_free(w);
  gsl_matrix_free(H);
  
  gsl_eigen_symmv_sort(*eval, *evec, GSL_EIGEN_SORT_VAL_ASC);
}

void eigen_norm_state_alloc(const gsl_matrix *evec, const double hstep, int state, gsl_vector_complex **psi_state)
{
  const int STATESIZE = evec->size1;
  if (evec->size2 != STATESIZE) {
    fprintf(stderr, "eigen_norm_state_alloc: evec not square");
    exit(1);
  }
  *psi_state = gsl_vector_complex_alloc(STATESIZE);
  double psi_norm = 0.0;
  for (int j = 0; j < STATESIZE; j++) {
    gsl_complex ej = gsl_complex_rect(gsl_matrix_get(evec, j, state), 0.0);
    gsl_vector_complex_set(*psi_state, j, ej);
    psi_norm += gsl_complex_abs2(ej) * hstep;
  }

  double sign = (gsl_matrix_get(evec, 1, state) > 0) ? 1.0 : (-1.0);
  gsl_vector_complex_scale(*psi_state, gsl_complex_rect(sign / sqrt(psi_norm), 0.0));
}
