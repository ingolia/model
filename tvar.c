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

#define NPTS 13
#define MIDDLE ((NPTS+1)/2)
#define STATESIZE (NPTS + 2)

#define MASS 1.0
#define HSTEP (1.0/8.0)
#define TSTEP (1.0/64.0)

int main(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_matrix *H1 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian(H0, V, MASS, HSTEP);
  gsl_vector_set(V, MIDDLE, 0.0);
  set_hamiltonian(H1, V, MASS, HSTEP);

  write_hamiltonian("tvardata/v0-H0", H0);
  write_hamiltonian("tvardata/v0-H1", H1);

  gsl_matrix_complex *U0 = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);
  
  FILE *fdebug = fopen("tvardata/v0-U.txt", "w");
  set_timeevol(U0, H0, H1, HSTEP, TSTEP, fdebug);
  check_unitarity(U0, fdebug);
  fclose(fdebug);
  
  gsl_vector_set(V, MIDDLE, 1.0);
  set_hamiltonian(H1, V, MASS, HSTEP);  

  write_hamiltonian("tvardata/v1-H0", H0);
  write_hamiltonian("tvardata/v1-H1", H1);
  
  gsl_matrix_complex *U1 = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);

  fdebug = fopen("tvardata/v1-U.txt", "w");
  set_timeevol(U1, H0, H1, HSTEP, TSTEP, fdebug);
  check_unitarity(U1, fdebug);
  fclose(fdebug);
  
  gsl_matrix_free(H0);
  gsl_matrix_free(H1);

  gsl_matrix_complex *DU = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);
  gsl_matrix_complex_memcpy(DU, U1);
  gsl_matrix_complex_sub(DU, U0);

  FILE *f = fopen("tvardata/DU.txt", "w");
  fwrite_matrix_complex(f, DU);
  fclose(f);
}
