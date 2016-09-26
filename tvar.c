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

#define NPTS 63
#define MIDDLE ((NPTS+1)/2)
#define STATESIZE (NPTS + 2)

#define MASS 1.0
#define HSTEP (1.0/64.0)
#define TSTEP (1.0/4096.0)

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

  iterate();
}

#define V0MAX 400.0
#define V0STEP 0.5

void iterate(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  gsl_matrix *Hprev = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_matrix *Hnext = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian(H0, V, MASS, HSTEP);
  gsl_matrix_complex *U0 = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);
  set_timeevol(U0, H0, H0, HSTEP, TSTEP, NULL);
  printf("U0           : ");
  check_unitarity(U0, NULL);

  gsl_matrix_complex *Utmp = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);
  gsl_matrix_complex *U0ttl = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);
  gsl_matrix_complex *U1ttl = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);

  gsl_matrix_complex_set_identity(U0ttl);
  gsl_matrix_complex_set_identity(U1ttl);
  
  gsl_complex one, zero;
  GSL_SET_COMPLEX(&one, 1.0, 0.0);
  GSL_SET_COMPLEX(&zero, 0.0, 0.0);
  
  for (double V0 = 0.0; V0 < V0MAX; V0 += V0STEP) {
    gsl_matrix_complex *U1 = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);

    gsl_vector_set(V, MIDDLE, V0);
    set_hamiltonian(Hprev, V, MASS, HSTEP);
    gsl_vector_set(V, MIDDLE, V0 + V0STEP);
    set_hamiltonian(Hnext, V, MASS, HSTEP);

    set_timeevol(U1, Hprev, Hnext, HSTEP, TSTEP, NULL);
    printf("U1    at %0.2f: ", V0);
    check_unitarity(U1, NULL);

    gsl_matrix_complex_memcpy(Utmp, U0ttl);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, one, U0, Utmp, zero, U0ttl);
    printf("U0ttl at %0.2f: ", V0);
    check_unitarity(U0ttl, NULL);
    
    gsl_matrix_complex_memcpy(Utmp, U1ttl);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, one, U1, Utmp, zero, U1ttl);
    printf("U1ttl at %0.2f: ", V0);
    check_unitarity(U1ttl, NULL);
  }

  gsl_matrix_complex *DU = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);
  gsl_matrix_complex_memcpy(DU, U1ttl);
  gsl_matrix_complex_sub(DU, U0ttl);

  FILE *f = fopen("tvardata/iterated.txt", "w");
  fwrite_matrix_complex(f, U0ttl);
  fwrite_matrix_complex(f, U1ttl);
  fwrite_matrix_complex(f, DU);
  fclose(f);  
}
