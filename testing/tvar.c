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
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_complex_double.h>

#include "schutil.h"
#include "writing.h"

#define NPTS 128
#define MIDDLE ((NPTS+1)/2)
#define STATESIZE (NPTS + 2)

#define PLANCK 4.0
#define MASS 1.0
#define HSTEP (1.0/64.0)
#define TSTEP (1.0/256.0)
#define WRITEEVERY 8

#define TFINAL 40.0

#define V0MAX 240.0

#define TSTART   2.0
#define THOLD    2.1
#define TRELEASE 4.1
#define TDONE    14.1

#define STATE0 2
#define STATE  3

void vtstep(gsl_vector *V, int tstep)
{
  vtstep_jump(V, tstep);
}

double vtscale(int tstep)
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

  return v0;
}

void vtstep_well(gsl_vector *V, int tstep)
{
  double v0 = vtscale(tstep);

  for (int i = 1; i < (V->size - 1); i++) {
    double dx = ((double) (i - MIDDLE)) / ((double) MIDDLE);
    gsl_vector_set(V, i, 0.5 * v0 * dx * dx);
  }
}

void vtstep_jump(gsl_vector *V, int tstep)
{
  double v0 = vtscale(tstep);

  for (int i = 1; i <= MIDDLE; i++) {
    gsl_vector_set(V, i, 0);
  }
  
  for (int i = MIDDLE + 1; i < (V->size - 1); i++) {
    gsl_vector_set(V, i, v0);
  }
}

void vtstep_force(gsl_vector *V, int tstep)
{
  double v0 = vtscale(tstep);
  
  double scale = 1.0 / ((double) V->size);
  
  for (int i = 1; i < (V->size - 1); i++) {
    gsl_vector_set(V, i, v0 * ((double) i) * scale);
  }
}

void stationary(void);
void evolve(void);

int main(void)
{
  evolve();
}

void stationary(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian(H0, V, PLANCK, MASS, HSTEP);

  gsl_vector *eval;
  gsl_matrix *evec;
  gsl_vector_complex *psi;
  
  eigen_solve_alloc(H0, &eval, &evec);

  FILE *f = fopen("tvardata/eig-v0-abs2.txt", "w");
  
  for (int j = STATE0; j < evec->size2; j++) {
    eigen_norm_state_alloc(evec, HSTEP, j, &psi);
    fprintf(f, "state%04d", (j - STATE));
    fwrite_vector_complex_abs2(f, psi);
    gsl_vector_complex_free(psi);
  }
  fclose(f);

  gsl_matrix_free(evec);
  gsl_vector_free(eval);
  
  vtstep(V, 1 + ((int) (THOLD / TSTEP)));
  set_hamiltonian(H0, V, PLANCK, MASS, HSTEP);
  eigen_solve_alloc(H0, &eval, &evec);

  f = fopen("tvardata/eig-v1-abs2.txt", "w");
  
  for (int j = STATE0; j < evec->size2; j++) {
    eigen_norm_state_alloc(evec, HSTEP, j, &psi);
    fprintf(f, "state%04d", (j - STATE));
    fwrite_vector_complex_abs2(f, psi);
    gsl_vector_complex_free(psi);
  }
  fclose(f);

  gsl_matrix_free(evec);
  gsl_vector_free(eval);
    
}

void evolve(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  gsl_matrix *Hprev = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_matrix *Hnext = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian(H0, V, PLANCK, MASS, HSTEP);

  gsl_vector *eval;
  gsl_matrix *evec;
  gsl_vector_complex *psi;
  gsl_vector_complex **psis = calloc(sizeof(gsl_vector_complex *), STATESIZE);
  
  eigen_solve_alloc(H0, &eval, &evec);
  for (int i = 0; i < STATESIZE; i++) {
    eigen_norm_state_alloc(evec, HSTEP, i, &(psis[i]));
  }
  
  eigen_norm_state_alloc(evec, HSTEP, STATE, &psi);

  FILE *psi1t = fopen("tvardata/psi-v1-t.txt", "w");
  
  timeevol_halves *U = timeevol_halves_alloc(STATESIZE);
  gsl_vector_complex *psinew = gsl_vector_complex_calloc(STATESIZE);
  gsl_vector_complex *psieig = gsl_vector_complex_calloc(STATESIZE);
  
  gsl_complex hstep_c;
  GSL_SET_COMPLEX(&hstep_c, HSTEP, 0.0);

  for (int tstep = 0; (tstep * TSTEP) <= TFINAL; tstep++) {
    const double t = tstep * TSTEP;

    vtstep(V, tstep);
    set_hamiltonian(Hprev, V, PLANCK, MASS, HSTEP);
    vtstep(V, tstep+1);
    set_hamiltonian(Hnext, V, PLANCK, MASS, HSTEP);

    set_timeevol_halves(U, Hprev, Hnext, PLANCK, TSTEP, NULL);
    timeevol_state(psinew, U, psi);

    if (tstep % WRITEEVERY == 0) {
      for (int i = 0; i < STATESIZE; i++) {
        gsl_blas_zdotc(psis[i], psi, gsl_vector_complex_ptr(psieig, i));
      }
      gsl_vector_complex_scale(psieig, hstep_c);

      printf("\033[2J\033[H");
      printf("t = %0.6f (tstep %6d)\n", t, tstep);
      terminal_graph_abs2(psi, 24, 2.0);
      puts("");
      terminal_graph_phase(psi, 8);
      puts("");
      terminal_graph_abs(psieig, 12, 1.2);
      
      fprintf(psi1t, "%0.6f", t);
      fwrite_vector_complex_abs2(psi1t, psi);
    }

    gsl_vector_complex_memcpy(psi, psinew);
  }

  fclose(psi1t);
}
