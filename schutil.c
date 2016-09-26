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

#include "schutil.h"

void check_deviation(const gsl_matrix *Z, size_t *imax, size_t *jmax, double *absmax, double *norm);

void set_hamiltonian(gsl_matrix *H, const gsl_vector *V, const double mass, const double hstep)
{
  const int npts = H->size1-2;

  if (H->size1 != H->size2) {
    fprintf(stderr, "set_hamiltonian: non-square H");
    exit(1);
  } else if (H->size1 != V->size) {
    fprintf(stderr, "set_hamiltonian: size of H and V don't match");
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

void set_timeevol(gsl_matrix_complex *U, const gsl_matrix *H0, const gsl_matrix *H1, const double hstep, const double tstep, FILE *fdebug)
{
  const int N = H0->size1;
  if (H0->size2 != N) {
    fprintf(stderr, "set_timeevol: non-square H0");
    exit(1);
  } else if (H1->size1 != N || H1->size2 != N) {
    fprintf(stderr, "set_timeevol: H1 does not match H0");
    exit(1);
  }
  
  gsl_matrix_complex *halfH = gsl_matrix_complex_alloc(N, N);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      gsl_complex Hij;
      GSL_SET_COMPLEX(&Hij, 0.25 * (gsl_matrix_get(H0, i, j) + gsl_matrix_get(H1, i, j)), 0.0);
      gsl_matrix_complex_set(halfH, i, j, Hij);
    }
  }

  if (fdebug) {
    fprintf(fdebug, "halfH = \n");
    fwrite_matrix_complex(fdebug, halfH);
  }
  
  gsl_complex ihdt;
  GSL_SET_COMPLEX(&ihdt, 0.0, 1.0 / tstep);
  
  gsl_matrix_complex *A = gsl_matrix_complex_alloc(N, N);
  gsl_matrix_complex_set_identity(A);
  gsl_matrix_complex_scale(A, ihdt);
  gsl_matrix_complex_sub(A, halfH);

  if (fdebug) {
    fprintf(fdebug, "A = \n");
    fwrite_matrix_complex(fdebug, A);
  }
  
  gsl_matrix_complex *Ainv = gsl_matrix_complex_alloc(N, N);
  gsl_permutation *Ap = gsl_permutation_alloc(N);
  int Asgn;
  gsl_linalg_complex_LU_decomp(A, Ap, &Asgn);
  gsl_linalg_complex_LU_invert(A, Ap, Ainv);
  gsl_permutation_free(Ap);

  gsl_matrix_complex *B = gsl_matrix_complex_alloc(N, N);
  gsl_matrix_complex_set_identity(B);
  gsl_matrix_complex_scale(B, ihdt);
  gsl_matrix_complex_add(B, halfH);

  gsl_complex one, zero;
  GSL_SET_COMPLEX(&one, 1.0, 0.0);
  GSL_SET_COMPLEX(&zero, 0.0, 0.0);
  gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, one, Ainv, B, zero, U);

  if (fdebug) {
    fprintf(fdebug, "Ainv = \n");
    fwrite_matrix_complex(fdebug, Ainv);
    fprintf(fdebug, "B = \n");    
    fwrite_matrix_complex(fdebug, B);
    fprintf(fdebug, "U = \n");    
    fwrite_matrix_complex(fdebug, U);
  } 
  
  gsl_matrix_complex_free(halfH);
  gsl_matrix_complex_free(A);
  gsl_matrix_complex_free(Ainv);
  gsl_matrix_complex_free(B);
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
  if (M->size1 != M->size2) {
    fprintf(stderr, "check_unitarity: non-square M");
    exit(1);
  }

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
