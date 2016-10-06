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

#define WRITEEVERY 8

#define FREE_TFINAL 1.0

#define STATE0 2
#define NSTATE 9

/*
 * For NPTS = 7
 *   |<- NPTS  ->|
 * 0 1 2 3 4 5 6 7 8
 * |<- STATESIZE ->|
 * At length = 1, HSTEP = 1/8 = LENGTH / (NPTS + 1)
 */

/* Simple infinite well of length l
 * psi_n = (2 / l) * exp(-i omega_n t) * sin (k_n x)
 * k_n = n * pi / l
 * E_n = n^2 * (pi^2 * planck^2) / (2 * m * l^2)
 * omega_n = n^2 * (pi^2 * planck) / (2 * m * l^2)
 * < psi_n | psi_m > = delta(m, n)
 */

void solve_stationary(const double planck, const double mass,
		  const int npts, const double length, const double tstep);

int main(void)
{
  solve_stationary(1.0, 1.0, 63, 2.0, 1.0 / 1024.0);

  solve_stationary(1.0, 0.5, 63, 2.0, 1.0 / 1024.0);
  solve_stationary(1.0, 2.0, 63, 2.0, 1.0 / 1024.0);

  solve_stationary(1.0, 1.0, 63, 1.0, 1.0 / 1024.0);
  solve_stationary(1.0, 1.0, 63, 4.0, 1.0 / 1024.0);

  solve_stationary(2.0, 1.0, 63, 2.0, 1.0 / 1024.0);
  solve_stationary(4.0, 1.0, 63, 2.0, 1.0 / 1024.0);
}

void solve_stationary(const double planck, const double mass,
		  const int npts, const double length, const double tstep)
{
  char *prefix, *filename;
  asprintf(&prefix, "statdata/psi-h%0.3f-m%0.3f-n%03d-l%0.3f-t%0.3f",
	 planck, mass, npts, length, tstep);

  const int statesize = npts + 2;
  const double hstep = length / (npts + 1);
  
  gsl_vector *V = gsl_vector_calloc(statesize);
  gsl_matrix *H0 = gsl_matrix_alloc(statesize, statesize);

  set_hamiltonian(H0, V, planck, mass, hstep);

  gsl_vector *eval;
  gsl_matrix *evec;

  gsl_vector_complex **psis;
  psis = malloc(sizeof(gsl_vector_complex *) * NSTATE);
  
  eigen_solve_alloc(H0, &eval, &evec);

  asprintf(&filename, "%s-eigstates.txt", prefix);
  FILE *f = fopen(filename, "w");
  free(filename);

  fprintf(f, "# planck = %0.6f\n", planck);
  fprintf(f, "# mass   = %0.6f\n", mass);
  fprintf(f, "# npts   = %6d\n", npts);
  fprintf(f, "# length = %0.6f\n", length);
  fprintf(f, "# tstep  = %0.6f\n", tstep);
  fprintf(f, "# |state|  %6d\n", statesize);
  fprintf(f, "# hstep  = %0.6f\n", hstep);

  fprintf(f, "n\tk\tEobs\tEcalc\tpsil2\n");
  
  for (int i = 0; (i < NSTATE); i++) {
    if ((i + STATE0) >= eval->size) {
      fprintf(stderr, "Not enough eigenstates for NSTATE");
      exit(1);
    }

    const int n = i + 1;
    double kn = n * M_PI / length;
    double En = n * n * M_PI * M_PI * planck * planck / (2.0 * mass * length * length);

    eigen_norm_state_alloc(evec, hstep, i + STATE0, &(psis[i]));

    asprintf(&filename, "%s-psi%03d.txt", prefix, n);
    FILE *fpsi = fopen(filename, "w");
    free(filename);
    fwrite_vector_complex_thorough(fpsi, psis[i]);
    fclose(fpsi);
    
    double l2 = 0.0;
    for (int j = 1; j < statesize; j++) {
      const double dev = (2.0 / length) * sin(kn * (j * hstep)) - GSL_REAL(gsl_vector_complex_get(psis[i], j));
      l2 += dev * dev * hstep;
    }
    fprintf(f, "%d\t%0.6f\t%0.6f\t%0.6f\t%0.6f\n",
	  n, kn, gsl_vector_get(eval, i + STATE0), En, l2);
  }
  
  fclose(f);
  
  FILE **psimagfiles = malloc(sizeof(FILE *) * NSTATE);
  FILE **psiphfiles = malloc(sizeof(FILE *) * NSTATE);
  
  for (int i = 0; i < NSTATE; i++) {
    asprintf(&filename, "%s-psi%03d-mag.txt", prefix, i+1);
    if ((psimagfiles[i] = fopen(filename, "w")) == NULL) {
      fprintf(stderr, "Failed to open \"%s\"\n", filename);
      exit(1);
    }
    free(filename);

    asprintf(&filename, "%s-psi%03d-ph.txt", prefix, i+1);
    if ((psiphfiles[i] = fopen(filename, "w")) == NULL) {
      fprintf(stderr, "Failed to open \"%s\"\n", filename);
      exit(1);
    }
    free(filename);
  }

  timeevol_halves *U0 = timeevol_halves_alloc(statesize);
  set_timeevol_halves(U0, H0, H0, planck, tstep, NULL);

  gsl_vector_complex *psinew = gsl_vector_complex_alloc(statesize);
  
  for (int t = 0; (t * tstep) <= FREE_TFINAL; t++) {
    const double time = t * tstep;

    if (t % WRITEEVERY == 0) {
      printf("time = %0.6f (t = %6d)\n", time, t);

      for (int i = 0; i < NSTATE; i++) {
        fprintf(psimagfiles[i], "%0.6f", time);
        fwrite_vector_complex_abs(psimagfiles[i], psis[i]);
        
        fprintf(psiphfiles[i], "%0.6f", time);
        fwrite_vector_complex_arg(psiphfiles[i], psis[i]);
      }
    }

    for (int i = 0; i < NSTATE; i++) {
      timeevol_state(psinew, U0, psis[i]);
      gsl_vector_complex_memcpy(psis[i], psinew);
    }
  }

  for (int i = 0; i < NSTATE; i++) {
    fclose(psimagfiles[i]);
    fclose(psiphfiles[i]);
  }

  terminal_graph_abs2(psis[0], 24, 2.0);
  terminal_graph_abs2(psis[1], 24, 2.0);
}
