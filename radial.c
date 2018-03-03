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

#define NPTS 512
#define STATESIZE (NPTS + 2)

#define PLANCK 4.0
#define MASS 1.0
#define HSTEP (1.0/32.0)
#define TSTEP (8.0/65536.0)
#define WRITEEVERY 256

#define TFINAL 40.0

#define V0 8.0

#define STATE0 1

#define OFFSET 32

void vtwell_one(gsl_vector *V)
{
  for (size_t i = 0; i < V->size; i++) {
    const double x = ((double) i) * HSTEP, x0 = ((double) NPTS + 1) / 2.0 * HSTEP, dx = fabs(x - x0);
    if (dx < HSTEP/2.01) { fprintf(stderr, "vtwell: |dx| = %0.3g < %0.3g\n", dx, HSTEP/2.01); exit(1); }
    gsl_vector_set(V, i, -V0 / dx);
  }
}

void vtwell_two(gsl_vector *V)
{
  for (size_t i = 0; i < V->size; i++) {
    const double x = ((double) i) * HSTEP;
    const double xa = ((double) NPTS + OFFSET + 1) / 2.0 * HSTEP;
    const double xb = ((double) NPTS - OFFSET + 1) / 2.0 * HSTEP;
    const double dxa = fabs(x - xa), dxb = fabs(x - xb);
    gsl_vector_set(V, i, (-V0 / dxa) + (-V0 / dxb));
  }
}

void evolve(void);

int main(void)
{
  evolve();
}

void evolve(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  vtwell_two(V);
  FILE *Vfile = fopen("radialdata/V.txt", "w");
  if (Vfile == NULL) { fprintf(stderr, "Failed to open V file\n"); exit(1); }
  fwrite_vector(Vfile, V);
  fclose(Vfile);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian(H0, V, PLANCK, MASS, HSTEP);

  gsl_vector *eval;
  gsl_matrix *evec;
  gsl_vector_complex *psi0;
  gsl_vector_complex **psis = calloc(sizeof(gsl_vector_complex *), STATESIZE);
  
  eigen_solve_alloc(H0, &eval, &evec);
  for (int i = 0; i < STATESIZE; i++) {
    eigen_norm_state_alloc(evec, HSTEP, i, &(psis[i]));
    char *filename;
    if (asprintf(&filename, "radialdata/psi-eig-%03d.txt", i) < 0) { exit(1); }
    FILE *eigi = fopen(filename, "w");
    if (eigi == NULL) { fprintf(stderr, "Failed to open %s\n", filename); exit(1); }
    fwrite_vector_complex_thorough(eigi, psis[i]);
    fclose(eigi);
  }
  
  eigen_norm_state_alloc(evec, HSTEP, STATE0, &psi0);

  FILE *psit = fopen("radialdata/psi-t.txt", "w");
  if (psit == NULL) { fprintf(stderr, "Failed to open output file\n"); exit(1); }
  
  timeevol_halves *U = timeevol_halves_alloc(STATESIZE);
  gsl_vector_complex *psinew = gsl_vector_complex_calloc(STATESIZE);
  gsl_vector_complex *psieig = gsl_vector_complex_calloc(STATESIZE);
  
  gsl_complex hstep_c;
  GSL_SET_COMPLEX(&hstep_c, HSTEP, 0.0);

  set_timeevol_halves(U, H0, H0, PLANCK, TSTEP, NULL);

  gsl_vector_complex *psi = gsl_vector_complex_alloc(STATESIZE);

  for (size_t i = 0; i < V->size; i++) {
    gsl_vector_complex_set(psi, i, gsl_vector_complex_get(psi0, i));
  }
  
  //  for (size_t i = 1; i <= DISPL; i++) {
  //    gsl_vector_complex_set(psi, i, gsl_vector_complex_get(psi0, 1));
  //  }
  //  for (size_t i = 1; i + DISPL + 1 < V->size; i++) {
  //    gsl_vector_complex_set(psi, i + DISPL, gsl_vector_complex_get(psi0, i));
  //  }
  
  for (int tstep = 0; (tstep * TSTEP) <= TFINAL; tstep++) {
    const double t = tstep * TSTEP;

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
      terminal_graph_abs(psi, 24, 2.0);
      puts("");
      terminal_graph_phase(psi, 8);
      puts("");
      terminal_graph_abs(psieig, 12, 1.2);
      
      fprintf(psit, "%0.6f", t);
      fwrite_vector_complex_abs2(psit, psi);
    }

    gsl_vector_complex_memcpy(psi, psinew);
  }

  fclose(psit);
}
