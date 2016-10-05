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
#include <gsl/gsl_vector_complex_double.h>

#include "schutil.h"
#include "writing.h"

#define NPTS 127
#define MIDDLE ((NPTS+1)/2)
#define STATESIZE (NPTS + 2)

#define PLANCK 1.0
#define MASS 1.0
#define HSTEP (1.0/64.0)
#define TSTEP (1.0/2048.0)

#define TFINAL 1.0

#define V0MAX 75.0

#define TSTART   0.5
#define THOLD    1.0
#define TRELEASE 1.5
#define TDONE    2.0

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

gsl_vector_complex *iterate_unitary(void);
gsl_vector_complex *iterate_solve(void);

int main(void)
{
  gsl_vector_complex *V_u = iterate_unitary();
  gsl_vector_complex *V_s = iterate_solve();

  fwrite_vector_complex_thorough(stderr, V_u);
  fwrite_vector_complex_thorough(stderr, V_s);

  gsl_vector_complex_sub(V_s, V_u);
  for (int j = 0; j < V_s->size; j++) {
    fprintf(stderr, "%03d %0.3e\n", j, gsl_complex_abs(gsl_vector_complex_get(V_s, j)));
  }
}

gsl_vector_complex *iterate_unitary(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  gsl_matrix *Hprev = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_matrix *Hnext = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian(H0, V, PLANCK, MASS, HSTEP);

  gsl_vector *eval;
  gsl_matrix *evec;
  gsl_vector_complex *psi0;
  
  eigen_solve_alloc(H0, &eval, &evec);

  eigen_norm_state_alloc(evec, GROUNDSTATE, &psi0);

  gsl_matrix_complex *Utmp = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);
  gsl_matrix_complex *U1ttl = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);

  gsl_matrix_complex_set_identity(U1ttl);
  
  gsl_complex one, zero;
  GSL_SET_COMPLEX(&one, 1.0, 0.0);
  GSL_SET_COMPLEX(&zero, 0.0, 0.0);
  
  for (int tstep = 0; (tstep * TSTEP) <= TFINAL; tstep++) {
    const double t = tstep * TSTEP;

    gsl_matrix_complex *U1 = gsl_matrix_complex_alloc(STATESIZE, STATESIZE);

    vtstep(V, tstep);
    set_hamiltonian(Hprev, V, PLANCK, MASS, HSTEP);
    vtstep(V, tstep+1);
    set_hamiltonian(Hnext, V, PLANCK, MASS, HSTEP);

    set_timeevol(U1, Hprev, Hnext, PLANCK, HSTEP, TSTEP, NULL);

    gsl_matrix_complex_memcpy(Utmp, U1ttl);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, one, U1, Utmp, zero, U1ttl);

    if (tstep % 64 == 0) {
      printf("U1ttl at %0.2f: ", t);
      check_unitarity(U1ttl, NULL);
    }
  }

  gsl_vector_complex *psit = gsl_vector_complex_alloc(STATESIZE);
  gsl_blas_zgemv(CblasNoTrans, one, U1ttl, psi0, zero, psit);

  gsl_vector_free(V);
  gsl_matrix_free(H0);
  gsl_matrix_free(Hprev);
  gsl_matrix_free(Hnext);
  gsl_vector_free(eval);
  gsl_matrix_free(evec);
  gsl_vector_complex_free(psi0);
  gsl_matrix_complex_free(Utmp);
  gsl_matrix_complex_free(U1ttl);

  return psit;
}

gsl_vector_complex *iterate_solve()
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  gsl_matrix *Hprev = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_matrix *Hnext = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian(H0, V, PLANCK, MASS, HSTEP);

  gsl_vector *eval;
  gsl_matrix *evec;
  gsl_vector_complex *psi0;
  
  eigen_solve_alloc(H0, &eval, &evec);

  eigen_norm_state_alloc(evec, GROUNDSTATE, &psi0);

  gsl_vector_complex *psiold = gsl_vector_complex_calloc(STATESIZE);
  gsl_vector_complex *psinew = gsl_vector_complex_calloc(STATESIZE); /* Returned */
  
  gsl_complex one, zero;
  GSL_SET_COMPLEX(&one, 1.0, 0.0);
  GSL_SET_COMPLEX(&zero, 0.0, 0.0);

  timeevol_halves *U = timeevol_halves_alloc(STATESIZE);

  gsl_vector_complex_memcpy(psiold, psi0);
  
  for (int tstep = 0; (tstep * TSTEP) <= TFINAL; tstep++) {
    const double t = tstep * TSTEP;

    vtstep(V, tstep);
    set_hamiltonian(Hprev, V, PLANCK, MASS, HSTEP);
    vtstep(V, tstep+1);
    set_hamiltonian(Hnext, V, PLANCK, MASS, HSTEP);

    set_timeevol_halves(U, Hprev, Hnext, PLANCK, TSTEP, NULL);
    timeevol_state(psinew, U, psiold);

    if (tstep % 64 == 0) {
      printf("Solving at %0.2f\n", t);
    }

    gsl_vector_complex_memcpy(psiold, psinew);
  }

  return psinew;
}

