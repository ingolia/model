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

#define NPTS 2048
#define STATESIZE (NPTS + 2)

#define PLANCK 4.0
#define MASS 1.0
#define HSTEP (8.0 / ((double) NPTS))
#define TSTEP (8.0/65536.0)
#define WRITEEVERY 256

#define TFINAL 40.0

#define V0 -16.0

#define STATE_MAX 4
#define STATE0 1

#define OFFSET 32

void v_rwell_one(gsl_vector *V, double xa, double Va) {
  for (size_t i = 0; i < V->size; i++) {
    const double x = ((double) i) * HSTEP;
    const double dx = fabs(x - xa);
    gsl_vector_set(V, i, Va / dx);
  }
}

void v_rwell_two(gsl_vector *V, double xa, double xb, double Va, double Vb) {
  for (size_t i = 0; i < V->size; i++) {
    const double x = ((double) i) * HSTEP;
    const double dxa = fabs(x - xa), dxb = fabs(x - xb);
    gsl_vector_set(V, i, (Va / dxa) + (Vb / dxb));

    if (0) {
      printf("%lu\t%0.6f\t%0.6f\t%0.6f\t%0.6f\n",
	   i, x, dxa, dxb, gsl_vector_get(V, i));
    }
  }
}

typedef struct {
  size_t noffset;
  double nucdist;
  double E0;
  double E1;
  double Enucrep;
  double Eelerep;
} spectrum_results;

void spectrum(spectrum_results *res, size_t offset);
void evolve(void);

int main(void)
{
  FILE *f = fopen("radialdata/distance-energetics.txt", "w");
  if (f == NULL) { fprintf(stderr, "Cannot open output data file\n"); exit(1); }
  
  spectrum_results specres;
  for (size_t offset = 2; offset < NPTS/4; offset += 2) {
    spectrum(&specres, offset);  
    fprintf(f, "%lu\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\n",
	  specres.noffset, specres.nucdist,
	  specres.E0, specres.E1, specres.Enucrep, specres.Eelerep);

    printf("%lu\t%0.6f\t%0.6f\t%0.6f\t%0.6f\t%0.6f\n",
	 specres.noffset, specres.nucdist,
	 specres.E0, specres.E1, specres.Enucrep, specres.Eelerep);
  }

  fclose(f);
}

void spectrum(spectrum_results *results, size_t noffset)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  char *prefix, *filename;
  if (asprintf(&prefix, "radialdata/spectrum-x%03lu", noffset) < 0) { exit(1); }

  const double xa = HSTEP * ((double) NPTS + (double) noffset + 1) / 2.0;
  const double xb = HSTEP * ((double) NPTS - (double) noffset + 1) / 2.0;

  results->noffset = noffset;
  results->nucdist = fabs(xb - xa);
  
  v_rwell_two(V, xa, xb, V0, V0);

  if (asprintf(&filename, "%s-V.txt", prefix) < 0) { exit(1); }
  FILE *Vfile = fopen(filename, "w");
  if (Vfile == NULL) { fprintf(stderr, "Failed to open V file %s\n", filename); exit(1); }
  free(filename);
  fwrite_vector(Vfile, V);
  fclose(Vfile);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian(H0, V, PLANCK, MASS, HSTEP);

  gsl_vector *eval;
  gsl_matrix *evec;
  gsl_vector_complex **psis = calloc(sizeof(gsl_vector_complex *), STATE_MAX);

  if (asprintf(&filename, "%s-E.txt", prefix) < 0) { exit (1); }
  FILE *Efile = fopen(filename, "w");
  if (Efile == NULL) { fprintf(stderr, "Failed to open E file %s\n", filename); exit(1); }
  free(filename);
  
  eigen_solve_alloc(H0, &eval, &evec);
  for (int i = 0; i < STATE_MAX; i++) {
    eigen_norm_state_alloc(evec, HSTEP, i, &(psis[i]));

    if (asprintf(&filename, "%s-psi-%02d.txt", prefix, i) < 0) { exit(1); }
    FILE *eigi = fopen(filename, "w");
    if (eigi == NULL) { fprintf(stderr, "Failed to open %s\n", filename); exit(1); }
    free(filename);
    fwrite_vector_complex_thorough(eigi, psis[i]);
    fclose(eigi);
    
    fprintf(Efile, "%d\t%0.3f\n", i, gsl_vector_get(eval, i));
  }  

  fclose(Efile);

  results->E0 = gsl_vector_get(eval, 0);
  results->E1 = gsl_vector_get(eval, 1);
  results->Enucrep = 2 * (-V0) / (xa - xb);

  double Eelerep = 0.0;
  for (size_t i = 0; i < psis[0]->size; i++) {
    for (size_t j = 0; j < psis[1]->size; j++) {
      const double xi = HSTEP * ((double) i);
      const double xj = HSTEP * ((double) j);
      const double dxij = 0.5 + fabs(xi - xj);
      const double Eij = (-V0) / dxij;
      const double occij = gsl_complex_abs2(gsl_vector_complex_get(psis[0], i))
        * gsl_complex_abs2(gsl_vector_complex_get(psis[1], j))
        * HSTEP * HSTEP;
      Eelerep += Eij * occij;
    }
  }
  results->Eelerep = Eelerep;
  
  if (0) {
    printf("\033[2J\033[H");
    
    gsl_vector_complex *psi0_disp;
    downsample_vector_complex_alloc(&psi0_disp, psis[0], 2);
    terminal_graph_abs2(psi0_disp, 24, 5.0);
    gsl_vector_complex_free(psi0_disp);
    puts("");
    downsample_vector_complex_alloc(&psi0_disp, psis[1], 2);
    terminal_graph_abs2(psi0_disp, 24, 5.0);
    gsl_vector_complex_free(psi0_disp);
  }

  gsl_vector_free(V);
  gsl_matrix_free(H0);
  gsl_vector_free(eval);
  gsl_matrix_free(evec);
  for (int i = 0; i < STATE_MAX; i++) {
    gsl_vector_complex_free(psis[i]);
  }
  free(psis);
  free(prefix);
}

void evolve(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  v_rwell_two(V,
	    HSTEP * ((double) NPTS + OFFSET + 1) / 2.0,
	    HSTEP * ((double) NPTS - OFFSET + 1) / 2.0,
	    V0, V0);
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
