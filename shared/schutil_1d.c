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
#include "schutil.h"
#include "schutil_1d.h"

void set_hamiltonian_bounded(gsl_matrix *H, 
			     const params *params,
			     const gsl_vector *V,
			     const double mass)
{
  const size_t npts = H->size1-2;

  ASSERT_SQUARE(H, "set_hamiltonian: H not square");
  ASSERT_SIZE1(H, V->size, "set_hamiltonian: dim(H) != dim(V)");
  
  gsl_matrix_set_all(H, 0.0);

  const double pfact = -0.5 * params->planck * params->planck / mass;
  const double hstep2 = 1.0 / (params->hstep * params->hstep);

  for (int j = 1; j <= npts; j++) {
    if (j > 1)    { gsl_matrix_set(H, j, j-1, pfact * hstep2); }
    if (j < npts) { gsl_matrix_set(H, j, j+1, pfact * hstep2); }

    gsl_matrix_set(H, j, j, -2.0 * pfact * hstep2 + gsl_vector_get(V, j));
  }
}

void set_hamiltonian_circular(gsl_matrix *H, 
			      const params *params,
			      const gsl_vector *V,
			      const double mass)
{
  const size_t npts = H->size1;

  ASSERT_SQUARE(H, "set_hamiltonian: H not square");
  ASSERT_SIZE1(H, V->size, "set_hamiltonian: dim(H) != dim(V)");
  
  gsl_matrix_set_all(H, 0.0);

  const double pfact = -0.5 * params->planck * params->planck / mass;
  const double hstep2 = 1.0 / (params->hstep * params->hstep);

  for (int j = 0; j < npts; j++) {
    if (j > 0) {
      gsl_matrix_set(H, j, j-1, pfact * hstep2);
    } else {
      gsl_matrix_set(H, j, npts-1, pfact * hstep2);
    }
    
    if (j + 1 < npts) {
      gsl_matrix_set(H, j, j+1, pfact * hstep2);
    } else {
      gsl_matrix_set(H, j, 0, pfact * hstep2);
    }

    gsl_matrix_set(H, j, j, -2.0 * pfact * hstep2 + gsl_vector_get(V, j));
  }
}
