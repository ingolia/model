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
#include "schdebug.h"
#include "timeevol.h"

timeevol *timeevol_alloc(const params *params)
{
  timeevol *U = malloc(sizeof(timeevol));
  U->B = gsl_matrix_complex_calloc(params->statesize, params->statesize);
  U->ALU = gsl_matrix_complex_calloc(params->statesize, params->statesize);
  U->ALUp = gsl_permutation_alloc(params->statesize);
  return U;
}

void timeevol_free(timeevol *U)
{
  gsl_permutation_free(U->ALUp);
  gsl_matrix_complex_free(U->ALU);
  gsl_matrix_complex_free(U->B);
  free(U);
}

/* Averaging Hs
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      gsl_complex *halfHij = gsl_matrix_complex_ptr(U->halfH, i, j);
      GSL_SET_COMPLEX(halfHij, 0.25 * (gsl_matrix_get(H0, i, j) + gsl_matrix_get(H1, i, j)), 0.0);
    }
  }

  if (fdebug) {
    fprintf(fdebug, "halfH = \n");
    fwrite_matrix_complex(fdebug, U->halfH);
  }
*/



void timeevol_set(timeevol *U,
		  const gsl_matrix_complex *Havg,
		  const params *params,
		  FILE *fdebug)
{
  const size_t N = params->statesize;

  ASSERT_SQUARE(Havg, "timeevol_halves: H0 not square");
  ASSERT_SQUARE_SIZE(Havg, N, "timeevol_halves: Havg bad size");
  ASSERT_SQUARE_SIZE(U->B, N, "timeevol_halves: U->B bad size");

  gsl_matrix_complex *A = gsl_matrix_complex_alloc(N, N);

  // A = ihdt I + 0.5 Havg = 0.5 * (2 ihdt I + Havg)
  gsl_matrix_complex_set_identity(A);
  gsl_matrix_complex_scale(A, gsl_complex_rect(0.0, 2.0 * params->planck / params->tstep));
  gsl_matrix_complex_sub(A, Havg);
  gsl_matrix_complex_scale(A, gsl_complex_rect(0.5, 0.0));

  if (fdebug) {
    fprintf(fdebug, "A = \n");
    fwrite_matrix_complex(fdebug, A);
  }

  int Asgn;
  gsl_matrix_complex_memcpy(U->ALU, A);
  gsl_linalg_complex_LU_decomp(U->ALU, U->ALUp, &Asgn);

  gsl_matrix_complex_free(A);

  gsl_matrix_complex_set_identity(U->B);
  gsl_matrix_complex_scale(U->B, gsl_complex_rect(0.0, 2.0 * params->planck / params->tstep));
  gsl_matrix_complex_add(U->B, Havg);
  gsl_matrix_complex_scale(U->B, gsl_complex_rect(0.5, 0.0)); 

  if (fdebug) {
    fprintf(fdebug, "B = \n");
    fwrite_matrix_complex(fdebug, U->B);
  }
}

void timeevol_state(gsl_vector_complex *psinew, const timeevol *U, const gsl_vector_complex *psiold)
{
  gsl_blas_zgemv(CblasNoTrans, gsl_complex_rect(1.0, 0.0), U->B, psiold, gsl_complex_rect(0.0, 0.0), psinew);    
  gsl_linalg_complex_LU_svx(U->ALU, U->ALUp, psinew);
}
