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
#define STATE1 3
#define STATE2 4

/*
 * For NPTS = 7
 *   |<- NPTS  ->|
 * 0 1 2 3 4 5 6 7 8
 * |<- STATESIZE ->|
 * At length = 1, HSTEP = 1/8 = LENGTH / (NPTS + 1)
 */

void solve_stationary(const double planck, const double mass,
		  const int npts, const double length, const double tstep);

int main(void)
{
  solve_stationary(1.0, 1.0, 255, 2.0, 1.0 / 1024.0);

  solve_stationary(1.0, 0.5, 255, 2.0, 1.0 / 1024.0);
  solve_stationary(1.0, 2.0, 255, 2.0, 1.0 / 1024.0);

  solve_stationary(1.0, 0.5, 255, 1.0, 1.0 / 1024.0);
  solve_stationary(1.0, 2.0, 255, 4.0, 1.0 / 1024.0);
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

  gsl_vector_complex *psi0, *psi1;
  
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
  
  for (int i = 0; (i + STATE0) < eval->size; i++) {
    const int n = i + 1;
    double kn = n * M_PI / length;
    double En = n * n * M_PI * M_PI * planck * planck / (2.0 * mass * length * length);

    gsl_vector_complex *psi_i;
    eigen_norm_state_alloc(evec, hstep, i + STATE0, &psi_i);

    asprintf(&filename, "%s-psi%03d.txt", prefix, n);
    FILE *fpsi = fopen(filename, "w");
    free(filename);
    fwrite_vector_complex_thorough(fpsi, psi_i);
    fclose(fpsi);
    
    double l2 = 0.0;
    for (int j = 1; j < statesize; j++) {
      const double dev = (2.0 / length) * sin(kn * (j * hstep)) - GSL_REAL(gsl_vector_complex_get(psi_i, j));
      l2 += dev * dev * hstep;
    }
    gsl_vector_complex_free(psi_i);

    fprintf(f, "%d\t%0.6f\t%0.6f\t%0.6f\t%0.6f\n",
	  n, kn, gsl_vector_get(eval, i + STATE0), En, l2);
  }
  
  fclose(f);
  
  eigen_norm_state_alloc(evec, hstep, STATE0, &psi0);
  eigen_norm_state_alloc(evec, hstep, STATE1, &psi1);
  
  FILE *psi0mag = fopen("tvardata/psi-v0-st0-mag.txt", "w");
  FILE *psi0ph = fopen("tvardata/psi-v0-st0-ph.txt", "w");
  FILE *psi1mag = fopen("tvardata/psi-v0-st1-mag.txt", "w");
  FILE *psi1ph = fopen("tvardata/psi-v0-st1-ph.txt", "w");

  timeevol_halves *U0 = timeevol_halves_alloc(statesize);
  set_timeevol_halves(U0, H0, H0, planck, tstep, NULL);

  gsl_vector_complex *psinew = gsl_vector_complex_alloc(statesize);
  
  for (int t = 0; (t * tstep) <= FREE_TFINAL; t++) {
    const double time = t * tstep;

    timeevol_state(psinew, U0, psi0);
    gsl_vector_complex_memcpy(psi0, psinew);

    timeevol_state(psinew, U0, psi1);
    gsl_vector_complex_memcpy(psi1, psinew);

    if (t % WRITEEVERY == 0) {
      printf("time = %0.6f (t = %6d)\n", time, t);

      fprintf(psi0mag, "%0.6f", time);
      fwrite_vector_complex_abs(psi0mag, psi0);

      fprintf(psi0ph, "%0.6f", time);
      fwrite_vector_complex_arg(psi0ph, psi0);

      fprintf(psi1mag, "%0.6f", time);
      fwrite_vector_complex_abs(psi1mag, psi1);
      
      fprintf(psi1ph, "%0.6f", time);
      fwrite_vector_complex_arg(psi1ph, psi1);
    }
  }

  fclose(psi0mag);
  fclose(psi0ph);
  fclose(psi1mag);
  fclose(psi1ph);

  terminal_graph_abs2(psi0, 24, 1.0/32.0);

  terminal_graph_abs2(psi1, 24, 1.0/32.0);
}
