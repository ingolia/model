#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <unistd.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_complex_double.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_complex_double.h>

#include "schutil.h"
#include "writing.h"

#define NPTS 64
#define STATESIZE NPTS

#define PLANCK 4.0
#define MASS 1.0
#define HSTEP (8.0 / ((double) NPTS))
#define TSTEP (8.0/65536.0)
#define WRITEEVERY 256

void solve_stationary(void);
void evolve(void);

int main(void)
{
  solve_stationary();
  evolve();
}

void solve_stationary()
{
  char *prefix, *filename;
  asprintf(&prefix, "circdata/psi-h%0.3f-m%0.3f-n%03d-h%0.3f-t%0.3f",
	 PLANCK, MASS, NPTS, HSTEP, TSTEP);

  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian_circular(H0, V, PLANCK, MASS, HSTEP);

  gsl_vector *eval;
  gsl_matrix *evec;

  gsl_vector_complex **psis;
  psis = malloc(sizeof(gsl_vector_complex *) * STATESIZE);
  
  eigen_solve_alloc(H0, &eval, &evec);

  double *ts = calloc(STATESIZE, sizeof(double));
  double **targs = calloc(STATESIZE, sizeof(double *));
  for (int i = 0; i < STATESIZE; i++) {
    targs[i] = calloc(STATESIZE, sizeof(double));
  }
  
  asprintf(&filename, "%s-eigstates.txt", prefix);
  FILE *f = fopen(filename, "w");
  free(filename);

  fprintf(f, "# planck = %0.6f\n", PLANCK);
  fprintf(f, "# mass   = %0.6f\n", MASS);
  fprintf(f, "# npts   = %6d\n", NPTS);
  fprintf(f, "# hstep = %0.6f\n", HSTEP);
  fprintf(f, "# tstep  = %0.6f\n", TSTEP);
  fprintf(f, "# |state|  %6d\n", STATESIZE);


  fprintf(f, "n\tEobs\n");

  for (int i = 0; (i < STATESIZE); i++) {
    eigen_norm_state_alloc(evec, HSTEP, i, &(psis[i]));

    asprintf(&filename, "%s-psi%03d.txt", prefix, i);
    FILE *fpsi = fopen(filename, "w");
    free(filename);
    fwrite_vector_complex_thorough(fpsi, psis[i]);
    fclose(fpsi);
    
    fprintf(f, "%d\t%0.6f\n",
	  i, gsl_vector_get(eval, i));
  }
  
  fclose(f);

  if (0) {
    for (int i = 0; i < 10; i++) {
      printf("\033[2J\033[H");
      printf("i = %d\n", i);
      terminal_graph_abs2(psis[i], 12, 0.5);
      puts("");
      terminal_graph_phase(psis[i], 8);
      sleep(3);
    }
  }
}

#define EVOLVE_STATE 1
#define TFINAL 10.0

void evolve(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  gsl_matrix *Hprev = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_matrix *Hnext = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian_circular(H0, V, PLANCK, MASS, HSTEP);

  gsl_vector *eval;
  gsl_matrix *evec;
  gsl_vector_complex *psi;
  gsl_vector_complex **psis = calloc(sizeof(gsl_vector_complex *), STATESIZE);
  
  eigen_solve_alloc(H0, &eval, &evec);
  for (int i = 0; i < STATESIZE; i++) {
    eigen_norm_state_alloc(evec, HSTEP, i, &(psis[i]));
  }

  psi = gsl_vector_complex_calloc(STATESIZE);
  
  for (int i = 0; i < STATESIZE; i++) {
    gsl_complex psi1 = gsl_vector_complex_get(psis[3], i);
    gsl_complex psia1 = gsl_complex_mul(psi1, gsl_complex_rect(M_SQRT1_2, 0.0));
    gsl_complex psi2 = gsl_vector_complex_get(psis[2], i);
    gsl_complex psia2 = gsl_complex_mul(psi2, gsl_complex_rect(0.0, M_SQRT1_2));
    
    gsl_vector_complex_set(psi, i, gsl_complex_add(psia1, psia2));
  }

  FILE *psi1t = fopen("circdata/psi-t.txt", "w");
  
  timeevol_halves *U = timeevol_halves_alloc(STATESIZE);
  gsl_vector_complex *psinew = gsl_vector_complex_calloc(STATESIZE);
  gsl_vector_complex *psieig = gsl_vector_complex_calloc(STATESIZE);
  
  gsl_complex hstep_c;
  GSL_SET_COMPLEX(&hstep_c, HSTEP, 0.0);

  for (int tstep = 0; (tstep * TSTEP) <= TFINAL; tstep++) {
    const double t = tstep * TSTEP;

    if (tstep % WRITEEVERY == 0) {
      for (int i = 0; i < STATESIZE; i++) {
        gsl_blas_zdotc(psis[i], psi, gsl_vector_complex_ptr(psieig, i));
      }
      gsl_vector_complex_scale(psieig, hstep_c);

      printf("\033[2J\033[H");
      printf("t = %0.6f (tstep %6d)\n", t, tstep);
      terminal_graph_abs2(psi, 20, 0.5);
      puts("");
      terminal_graph_phase(psi, 8);
      puts("");
      terminal_graph_abs(psieig, 12, 1.2);
      
      fprintf(psi1t, "%0.6f", t);
      fwrite_vector_complex_abs2(psi1t, psi);
    }

    set_hamiltonian_circular(Hprev, V, PLANCK, MASS, HSTEP);
    set_hamiltonian_circular(Hnext, V, PLANCK, MASS, HSTEP);

    set_timeevol_halves(U, Hprev, Hnext, PLANCK, TSTEP, NULL);
    timeevol_state(psinew, U, psi);

    gsl_vector_complex_memcpy(psi, psinew);
  }

  fclose(psi1t);
}
