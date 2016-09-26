#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <unistd.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_complex_double.h>
#include <gsl/gsl_vector.h>

#include "schutil.h"

#define NPTS 255
#define MIDDLE ((NPTS+1)/2)
#define STATESIZE (NPTS + 2)

#define MASS 1.0
#define HSTEP (1.0/64.0)

void solve_potential(const char *prefix, const gsl_vector *V);

int main(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  solve_potential("data/none", V);

  char *prefix;

  for (double V0 = 0.5; V0 < 1025; V0 *= 2.0) {
    if (asprintf(&prefix, "data/%0.1f", V0) < 0) { exit(1); }

    gsl_vector_set(V, MIDDLE, V0);
    solve_potential(prefix, V);

    free(prefix);
  }
}

void solve_potential(const char *prefix, const gsl_vector *V)
{
  write_potential(prefix, V);

  gsl_matrix *H = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian(H, V, MASS, HSTEP);
  check_symmetry(H);
  write_hamiltonian(prefix, H);

  gsl_vector *eval = gsl_vector_alloc(STATESIZE);
  gsl_matrix *evec = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(STATESIZE);

  gsl_eigen_symmv(H, eval, evec, w);

  gsl_eigen_symmv_free(w);

  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_ASC);

  write_energies(prefix, evec, eval);
  write_psi(prefix, evec, eval);
}


