#define _GNU_SOURCE
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
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

#include "schdebug.h"

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

void write_hamiltonian(const char *prefix, const gsl_matrix *H)
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
    fprintf(fout, "\tk%d", k);
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

void fwrite_vector(FILE *f, const gsl_vector *V)
{
  for (int j = 0; j < V->size; j++) {
    fprintf(f, "%0.6f\n", gsl_vector_get(V, j));
  }
}

void fwrite_vector_complex_thorough(FILE *f, const gsl_vector_complex *V)
{
  for (int j = 0; j < V->size; j++) {
    gsl_complex Vj = gsl_vector_complex_get(V, j);
    
    fprintf(f, "%4d %0.4f+%0.4fi  %0.6f  %0.3f\n", 
	  j, GSL_REAL(Vj), GSL_IMAG(Vj),
	  gsl_complex_abs(Vj), gsl_complex_arg(Vj));
  }
}

void fwrite_matrix(FILE *f, const gsl_matrix *M)
{
  fprintf(f, "j");
  for (int k = 0; k < M->size2; k++) {
    fprintf(f, "\tk%d", k);
  }
  fprintf(f, "\n");

  for (int j = 0; j < M->size1; j++) {
    fprintf(f, "%d", j);

    for (int k = 0; k < M->size2; k++) {
      fprintf(f, "\t%0.4f", gsl_matrix_get(M, j, k));
    }

    fprintf(f, "\n");
  }
}

void fwrite_matrix_complex(FILE *f, const gsl_matrix_complex *M)
{
  fprintf(f, "j");
  for (int k = 0; k < M->size2; k++) {
    fprintf(f, "\tk%d", k);
  }
  fprintf(f, "\n");

  for (int j = 0; j < M->size1; j++) {
    fprintf(f, "%d", j);

    for (int k = 0; k < M->size2; k++) {
      const gsl_complex Mjk = gsl_matrix_complex_get(M, j, k);
      fprintf(f, "\t%0.4f+%0.4fi", GSL_REAL(Mjk), GSL_IMAG(Mjk));
    }

    fprintf(f, "\n");
  }
}

void fwrite_vector_complex_abs2(FILE *f, const gsl_vector_complex *psi)
{
  for (int j = 0; j < psi->size; j++) {
    fprintf(f, "\t%0.6f", gsl_complex_abs2(gsl_vector_complex_get(psi, j)));
  }
  fprintf(f, "\n");
}

void fwrite_vector_complex_abs(FILE *f, const gsl_vector_complex *psi)
{
  for (int j = 0; j < psi->size; j++) {
    fprintf(f, "\t%0.6f", gsl_complex_abs(gsl_vector_complex_get(psi, j)));
  }
  fprintf(f, "\n");
}

void fwrite_vector_complex_arg(FILE *f, const gsl_vector_complex *psi)
{
  for (int j = 0; j < psi->size; j++) {
    fprintf(f, "\t%0.3f", gsl_complex_arg(gsl_vector_complex_get(psi, j)));
  }
  fprintf(f, "\n");
}

void check_symmetry(const gsl_matrix *M)
{
  if (M->size1 != M->size2) {
    fprintf(stderr, "check_hermiticity: non-square M");
    exit(1);
  }

  gsl_matrix *Mt = gsl_matrix_alloc(M->size2, M->size1);
  gsl_matrix_transpose_memcpy(Mt, M);
  gsl_matrix_sub(Mt, M);

  size_t imax, jmax;
  double absmax, norm;
  check_deviation(Mt, &imax, &jmax, &absmax, &norm);
  printf("check_symmetry: |M - Mt|^2 = %e, max at (%lu, %lu) = %e\n", norm, imax, jmax, absmax);
  
  gsl_matrix_free(Mt);
}

void check_unitarity(const gsl_matrix_complex *M, FILE *fdebug)
{
  ASSERT_SQUARE(M, "check_unitarity: not square");
  
  gsl_matrix_complex *Mt = gsl_matrix_complex_alloc(M->size2, M->size1);
  for (int i = 0; i < M->size1; i++) {
    for (int j = 0; j < M->size2; j++) {
      gsl_complex Mijc = gsl_complex_conjugate(gsl_matrix_complex_get(M, i, j));
      gsl_matrix_complex_set(Mt, j, i, Mijc);
    }
  }

  gsl_matrix_complex *MtM = gsl_matrix_complex_alloc(M->size2, M->size2);
  gsl_matrix_complex_set_identity(MtM);
  
  gsl_complex one, negone;
  GSL_SET_COMPLEX(&one, 1.0, 0.0);
  GSL_SET_COMPLEX(&negone, -1.0, 0.0);
  gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, one, Mt, M, negone, MtM);

  size_t imax, jmax;
  gsl_complex absmax;
  GSL_SET_COMPLEX(&absmax, 0.0, 0.0);
  double norm = 0.0;
  for (size_t i = 0; i < MtM->size1; i++) {
    for (size_t j = 0; j < MtM->size2; j++){
      gsl_complex Zij = gsl_matrix_complex_get(MtM, i, j);
      if (gsl_complex_abs2(Zij) > gsl_complex_abs2(absmax)) {
        absmax = Zij;
        imax = i;
        jmax = j;
      }

      norm += gsl_complex_abs2(Zij);
    }
  }
  printf("check_unitarity: |MtM - I| = %e, max at (%lu, %lu) = %0.4f+%0.4fi\n", norm, imax, jmax, GSL_REAL(absmax), GSL_IMAG(absmax));
  
  gsl_matrix_complex_free(Mt);
  gsl_matrix_complex_free(MtM);
}

void check_deviation(const gsl_matrix *Z, size_t *imax, size_t *jmax, double *absmax, double *norm)
{
  (*imax) = 0;
  (*jmax) = 0;
  (*absmax) = -1;
  (*norm) = 0.0;
  for (size_t i = 0; i < Z->size1; i++) {
    for (size_t j = 0; j < Z->size2; j++){
      const double Zij = gsl_matrix_get(Z, i, j);
      if (fabs(Zij) > (*absmax)) {
        (*absmax) = fabs(Zij);
        (*imax) = i;
        (*jmax) = j;
      }

      (*norm) += (Zij * Zij);
    }
  }
}

FILE *fopenf(const char *mode, const char *fmt, ...)
{
  char *fname;

  va_list ap;
  va_start(ap, fmt);
  if (vasprintf(&fname, fmt, ap) < 0) {
    fprintf(stderr, "Unable to format filename\n");
    exit(1);
  }
  va_end(ap);

  FILE *f = fopen(fname, mode);
  if (f == NULL) {
    fprintf(stderr, "Unable to open \"%s\"\n", fname);
    exit(1);
  }

  free(fname);

  return f;
}
