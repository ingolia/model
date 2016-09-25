#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <unistd.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "schutil.h"

void set_hermitian(gsl_matrix *H, const gsl_vector *V, const double mass, const double hstep)
{
  const int npts = H->size1-2;

  if (H->size1 != H->size2) {
    fprintf(stderr, "set_hermitian: non-square H");
    exit(1);
  } else if (H->size1 != V->size) {
    fprintf(stderr, "set_hermitian: size of H and V don't match");
    exit(1);
  }
  
  gsl_matrix_set_all(H, 0.0);

  for (int j = 1; j <= npts; j++) {
    const double pfact = -0.5 / mass;
    const double hstep2 = 1.0 / (hstep * hstep);

    if (j > 1)    { gsl_matrix_set(H, j, j-1, pfact * hstep2); }
    if (j < npts) { gsl_matrix_set(H, j, j+1, pfact * hstep2); }

    gsl_matrix_set(H, j, j, -2.0 * pfact * hstep2 + gsl_vector_get(V, j));
  }
}

void write_potential(const char *prefix, const gsl_vector *V)
{
  char *stname;
  if (asprintf(&stname, "%s_V.txt", prefix) < 0) { exit(1); }
  FILE *fpot;
  if ((fpot = fopen(stname, "w")) == NULL) {
    fprintf(stderr, "Cannot open output file \"%s\"\n", stname);
    exit(1);
  }
  free(stname);

  for (int j = 1; j < (V->size - 1); j++) {
    fprintf(fpot, "%d\t%0.6f\n", j, gsl_vector_get(V, j));
  }

  fclose(fpot);
}

void write_hermitian(const char *prefix, const gsl_matrix *H)
{
  FILE *fout;

  char *stname;
  if (asprintf(&stname, "%s_H.txt", prefix) < 0) { exit(1); }
  if ((fout = fopen(stname, "w")) == NULL) {
    fprintf(stderr, "Cannot open output file \"%s\"\n", stname);
    exit(1);
  }
  free(stname);

  fprintf(fout, "j");
  for (int k = 0; k < H->size2; k++) {
    fprintf(fout, "\tk_%d", k);
  }
  fprintf(fout, "\n");
  
  for (int j = 0; j < H->size1; j++) {
    fprintf(fout, "%d", j);

    for (int k = 0; k < H->size2; k++) {
      fprintf(fout, "\t%0.6f", gsl_matrix_get(H, j, k));
    }

    fprintf(fout, "\n");
  }

  
  fclose(fout);
}

void write_energies(const char *prefix, const gsl_matrix *Psis, const gsl_vector *Es)
{
  FILE *fout;

  char *stname;
  if (asprintf(&stname, "%s_E.txt", prefix) < 0) { exit(1); }
  if ((fout = fopen(stname, "w")) == NULL) {
    fprintf(stderr, "Cannot open output file \"%s\"\n", stname);
    exit(1);
  }
  free(stname);

  fprintf(fout, "phi_n\tE_n\n");
  for (int st = 0; st < Es->size; st++) {
    fprintf(fout, "%d\t%0.6f", st, gsl_vector_get(Es, st));      
    fprintf(fout, "\n");
  }

  fclose(fout);
}

void write_psi(const char *prefix, const gsl_matrix *Psis, const gsl_vector *Es)
{
  FILE *fout;

  char *stname;
  if (asprintf(&stname, "%s_psi.txt", prefix) < 0) { exit(1); }
  if ((fout = fopen(stname, "w")) == NULL) {
    fprintf(stderr, "Cannot open output file \"%s\"\n", stname);
    exit(1);
  }
  free(stname);

  fprintf(fout, "j");
  for (int st = 0; st < Es->size; st++) {
    fprintf(fout, "\tpsi_%d", st);
  }
  fprintf(fout, "\n");
  
  for (int j = 0; j < Psis->size1; j++) {
    fprintf(fout, "%d", j);

    for (int st = 0; st < Psis->size2; st++) {
      double psi_st_j = gsl_matrix_get(Psis, j, st);
      fprintf(fout, "\t%0.6f", psi_st_j);
    }

    fprintf(fout, "\n");
  }

  
  fclose(fout);
}

