#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <unistd.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_complex_double.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_complex_double.h>

#include "schutil.h"

#define NPTS 127
#define MIDDLE ((NPTS+1)/2)
#define STATESIZE (NPTS + 2)

#define MASS 1.0
#define HSTEP (1.0/64.0)
#define TSTEP (1.0/1024.0)

void iterate(void);
void fwrite_evolved_psi(FILE *f, const gsl_vector_complex *psi0, const gsl_matrix_complex *Ut0t);

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

#define TFINAL 5

#define V0MAX 50.0

#define TSTART   0.5
#define THOLD    1.5
#define TRELEASE 2.5
#define TDONE    3.5

#define GROUNDSTATE 2

void vtstep(gsl_vector *V, int tstep)
{
  const double t = tstep * TSTEP;

  double v0;
  if (t < TSTART) {
    v0 = 0;
  } else if (t < THOLD) {
    v0 = V0MAX * ((t - TSTART) / (THOLD - TSTART));
  } else if (t < TRELEASE) {
    v0 = V0MAX;
  } else if (t < TDONE) {
    v0 = V0MAX * ((TDONE - t) / (TDONE - TRELEASE));
  } else {
    v0 = 0.0;
  }

  double scale = 1.0 / ((double) V->size);
  
  for (int i = 1; i < (V->size - 1); i++) {
    gsl_vector_set(V, i, v0 * ((double) i) * scale);
  }
}

void iterate(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  gsl_matrix *Hprev = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_matrix *Hnext = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian(H0, V, MASS, HSTEP);

  /* Extract ground state into complex \Psi */
  gsl_vector *eval = gsl_vector_alloc(STATESIZE);
  gsl_matrix *evec = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(STATESIZE);

  gsl_eigen_symmv(H0, eval, evec, w);

  gsl_eigen_symmv_free(w);

  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_ASC);

  gsl_vector_complex *psi0 = gsl_vector_complex_alloc(STATESIZE);
  double psi0norm = 0.0;
  for (int j = 0; j < STATESIZE; j++) {
    gsl_complex ej = gsl_complex_rect(gsl_matrix_get(evec, j, GROUNDSTATE), 0.0);
    gsl_vector_complex_set(psi0, j, ej);
    psi0norm += gsl_complex_abs2(ej);
  }
  gsl_vector_complex_scale(psi0, gsl_complex_rect(1.0 / sqrt(psi0norm), 0.0));

  FILE *psi0t = fopen("tvardata/psi-v0-t.txt", "w");
  FILE *psi1t = fopen("tvardata/psi-v1-t.txt", "w");
  
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
  
  for (int tstep = 0; (tstep * TSTEP) <= TFINAL; tstep++) {
    const double t = tstep * TSTEP;

    gsl_matrix_complex *U1 = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);

    vtstep(V, tstep);
    set_hamiltonian(Hprev, V, MASS, HSTEP);
    vtstep(V, tstep+1);
    set_hamiltonian(Hnext, V, MASS, HSTEP);

    set_timeevol(U1, Hprev, Hnext, HSTEP, TSTEP, NULL);

    gsl_matrix_complex_memcpy(Utmp, U0ttl);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, one, U0, Utmp, zero, U0ttl);
    printf("U0ttl at %0.2f: ", tstep*TSTEP);
    check_unitarity(U0ttl, NULL);

    fprintf(psi0t, "%0.6f", tstep*TSTEP);
    fwrite_evolved_psi(psi0t, psi0, U0ttl);
    
    gsl_matrix_complex_memcpy(Utmp, U1ttl);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, one, U1, Utmp, zero, U1ttl);
    printf("U1ttl at %0.2f: ", tstep*TSTEP);
    check_unitarity(U1ttl, NULL);

    fprintf(psi1t, "%0.6f", tstep*TSTEP);
    fwrite_evolved_psi(psi1t, psi0, U1ttl);
  }

  gsl_matrix_complex *DU = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);
  gsl_matrix_complex_memcpy(DU, U1ttl);
  gsl_matrix_complex_sub(DU, U0ttl);

  FILE *f = fopen("tvardata/iterated.txt", "w");
  fwrite_matrix_complex(f, U0ttl);
  fwrite_matrix_complex(f, U1ttl);
  fwrite_matrix_complex(f, DU);
  fclose(f);  

  fclose(psi0t);
  fclose(psi1t);
}

void fwrite_evolved_psi(FILE *f, const gsl_vector_complex *psi0, const gsl_matrix_complex *Ut0t)
{
  gsl_complex one, zero;
  GSL_SET_COMPLEX(&one, 1.0, 0.0);
  GSL_SET_COMPLEX(&zero, 0.0, 0.0);

  gsl_vector_complex *psit = gsl_vector_complex_alloc(STATESIZE);
  gsl_blas_zgemv(CblasNoTrans, one, Ut0t, psi0, zero, psit);

  for (int j = 0; j < STATESIZE; j++) {
    fprintf(f, "\t%0.6f", gsl_complex_abs2(gsl_vector_complex_get(psit, j)));
  }
  fprintf(f, "\n");

  gsl_vector_complex_free(psit);
}
