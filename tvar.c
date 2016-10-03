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

#define NPTS 127
#define MIDDLE ((NPTS+1)/2)
#define STATESIZE (NPTS + 2)

#define MASS 1.0
#define HSTEP (1.0/64.0)
#define TSTEP (1.0/1024.0)
#define WRITEEVERY 8

void unperturbed(void);
void iterate(void);

void fwrite_evolved_psi(FILE *f, const gsl_vector_complex *psi);
void fwrite_evolved_psi_magnitude(FILE *f, const gsl_vector_complex *psi);
void fwrite_evolved_psi_phase(FILE *f, const gsl_vector_complex *psi);

int main(void)
{
  unperturbed();
  iterate();
}

#define FREE_TFINAL 5.0

#define STATE0 2
#define STATE1 3
#define STATE2 4

void unperturbed(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian(H0, V, MASS, HSTEP);

  gsl_vector *eval;
  gsl_matrix *evec;

  gsl_vector_complex *psi0, *psi1;
  
  eigen_solve_alloc(H0, &eval, &evec);

  eigen_norm_state_alloc(evec, STATE0, &psi0);
  eigen_norm_state_alloc(evec, STATE1, &psi1);
  
  FILE *psi0mag = fopen("tvardata/psi-v0-st0-mag.txt", "w");
  FILE *psi0ph = fopen("tvardata/psi-v0-st0-ph.txt", "w");
  FILE *psi1mag = fopen("tvardata/psi-v0-st1-mag.txt", "w");
  FILE *psi1ph = fopen("tvardata/psi-v0-st1-ph.txt", "w");

  timeevol_halves *U0 = timeevol_halves_alloc(STATESIZE);
  set_timeevol_halves(U0, H0, H0, TSTEP, NULL);

  gsl_vector_complex *psinew = gsl_vector_complex_alloc(STATESIZE);
  
  for (int tstep = 0; (tstep * TSTEP) <= FREE_TFINAL; tstep++) {
    const double t = tstep * TSTEP;

    timeevol_state(psinew, U0, psi0);
    gsl_vector_complex_memcpy(psi0, psinew);

    timeevol_state(psinew, U0, psi1);
    gsl_vector_complex_memcpy(psi1, psinew);

    if (tstep % WRITEEVERY == 0) {
      printf("t = %0.6f (tstep %6d)\n", t, tstep);

      fprintf(psi0mag, "%0.6f", t);
      fwrite_evolved_psi_magnitude(psi0mag, psi0);

      fprintf(psi0ph, "%0.6f", t);
      fwrite_evolved_psi_phase(psi0ph, psi0);

      fprintf(psi1mag, "%0.6f", t);
      fwrite_evolved_psi_magnitude(psi1mag, psi1);
      
      fprintf(psi1ph, "%0.6f", t);
      fwrite_evolved_psi_phase(psi1ph, psi1);
    }
  }

  fclose(psi0mag);
  fclose(psi0ph);
  fclose(psi1mag);
  fclose(psi1ph);

  terminal_graph_abs2(psi0, 24, 1.0/32.0);

  terminal_graph_abs2(psi1, 24, 1.0/32.0);
}

#define TFINAL 4.0

#define V0MAX 5.0

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

void iterate(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  gsl_matrix *Hprev = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_matrix *Hnext = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian(H0, V, MASS, HSTEP);

  gsl_vector *eval;
  gsl_matrix *evec;
  gsl_vector_complex *psi;
  
  eigen_solve_alloc(H0, &eval, &evec);

  eigen_norm_state_alloc(evec, STATE0, &psi);

  FILE *psi1t = fopen("tvardata/psi-v1-t.txt", "w");
  
  timeevol_halves *U = timeevol_halves_alloc(STATESIZE);
  gsl_vector_complex *psinew = gsl_vector_complex_calloc(STATESIZE);
  
  for (int tstep = 0; (tstep * TSTEP) <= TFINAL; tstep++) {
    const double t = tstep * TSTEP;

    vtstep(V, tstep);
    set_hamiltonian(Hprev, V, MASS, HSTEP);
    vtstep(V, tstep+1);
    set_hamiltonian(Hnext, V, MASS, HSTEP);

    set_timeevol_halves(U, Hprev, Hnext, TSTEP, NULL);
    timeevol_state(psinew, U, psi);

    if (tstep % WRITEEVERY == 0) {
      printf("\033[2J\033[H");
      printf("t = %0.6f (tstep %6d)\n", t, tstep);
      terminal_graph_abs2(psi, 24, 1.0/16.0);
      puts("");
      terminal_graph_phase(psi, 8);
      
      fprintf(psi1t, "%0.6f", t);
      fwrite_evolved_psi(psi1t, psi);
    }

    gsl_vector_complex_memcpy(psi, psinew);
  }

  fclose(psi1t);
}

void fwrite_evolved_psi(FILE *f, const gsl_vector_complex *psi)
{
  for (int j = 0; j < STATESIZE; j++) {
    fprintf(f, "\t%0.6f", gsl_complex_abs2(gsl_vector_complex_get(psi, j)));
  }
  fprintf(f, "\n");
}

void fwrite_evolved_psi_magnitude(FILE *f, const gsl_vector_complex *psi)
{
  for (int j = 0; j < STATESIZE; j++) {
    fprintf(f, "\t%0.6f", gsl_complex_abs(gsl_vector_complex_get(psi, j)));
  }
  fprintf(f, "\n");
}

void fwrite_evolved_psi_phase(FILE *f, const gsl_vector_complex *psi)
{
  for (int j = 0; j < STATESIZE; j++) {
    fprintf(f, "\t%0.3f", gsl_complex_arg(gsl_vector_complex_get(psi, j)));
  }
  fprintf(f, "\n");
}
