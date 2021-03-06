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

#include "grid2d.h"
#include "schutil.h"
#include "writing.h"

void check_deviation(const gsl_matrix *Z, size_t *imax, size_t *jmax, double *absmax, double *norm);

void set_hamiltonian(gsl_matrix *H, const gsl_vector *V, const double planck, const double mass, const double hstep)
{
  const size_t npts = H->size1-2;

  ASSERT_SQUARE(H, "set_hamiltonian: H not square");
  ASSERT_SIZE1(H, V->size, "set_hamiltonian: dim(H) != dim(V)");
  
  gsl_matrix_set_all(H, 0.0);

  const double pfact = -0.5 * planck * planck / mass;
  const double hstep2 = 1.0 / (hstep * hstep);

  for (int j = 1; j <= npts; j++) {
    if (j > 1)    { gsl_matrix_set(H, j, j-1, pfact * hstep2); }
    if (j < npts) { gsl_matrix_set(H, j, j+1, pfact * hstep2); }

    gsl_matrix_set(H, j, j, -2.0 * pfact * hstep2 + gsl_vector_get(V, j));
  }
}

void set_hamiltonian_circular(gsl_matrix *H, const gsl_vector *V, const double planck, const double mass, const double hstep)
{
  const size_t npts = H->size1;

  ASSERT_SQUARE(H, "set_hamiltonian: H not square");
  ASSERT_SIZE1(H, V->size, "set_hamiltonian: dim(H) != dim(V)");
  
  gsl_matrix_set_all(H, 0.0);

  const double pfact = -0.5 * planck * planck / mass;
  const double hstep2 = 1.0 / (hstep * hstep);

  for (int j = 0; j < npts; j++) {
    if (j > 0) {
      gsl_matrix_set(H, j, j-1, pfact * hstep2);
    } else {
      gsl_matrix_set(H, j, npts-1, pfact * hstep2);
    }
    
    if (j + 1 < npts) {
      gsl_matrix_set(H, j, j+1, pfact * hstep2);
    } else {
      gsl_matrix_set(H, j, 0, pfact * hstep2);
    }

    gsl_matrix_set(H, j, j, -2.0 * pfact * hstep2 + gsl_vector_get(V, j));
  }
}

void set_hamiltonian_sq2d(gsl_matrix *H, const gsl_vector *V, const double planck, const double mass, const double hstep, const grid2d *grid)
{
  ASSERT_SQUARE_SIZE(H, grid->npts, "set_hamiltonian: dim(H) != |grid|");
  ASSERT_SIZE1(H, V->size, "set_hamiltonian: dim(H) != dim(V)");

  grid2d_set_laplacian(H, grid); /* The POSITIVE laplacian */

  const double pfact = -0.5 * planck * planck / mass;
  const double hstep2 = 1.0 / (hstep * hstep);

  gsl_matrix_scale(H, -1.0 * pfact * hstep2);

  for (size_t i = 0; i < grid->npts; i++) {
    (*gsl_matrix_ptr(H, i, i)) += gsl_vector_get(V, i);
  }
}

// <psi' | H | psi>
// Take psi = a + bi for a, b real and psi' = a - bi
// <psi' | H | a+bi> = <psi' | H | a> + i <psi' | H | b>
//                   = <a|H|a> - i <b|H|a> + i <a|H|b> + i (-i <b|H|b>)
// H symmetric, <b|H|a> = <a|H|b> and kill imaginary terms
//                   = <a|H|a> + <b|H|b>
double get_energy(const gsl_matrix *H, const double hstep, const gsl_vector_complex *psi)
{
  gsl_vector_const_view a = gsl_vector_complex_const_real(psi);
  gsl_vector_const_view b = gsl_vector_complex_const_imag(psi);

  gsl_vector *q = gsl_vector_alloc(psi->size);
  double aHa;
  gsl_blas_dsymv(CblasUpper, 1.0, H, &(a.vector), 0.0, q);
  gsl_blas_ddot(&(a.vector), q, &aHa);

  double bHb;
  gsl_blas_dsymv(CblasUpper, 1.0, H, &(b.vector), 0.0, q);
  gsl_blas_ddot(&(b.vector), q, &bHb);
  
  gsl_vector_free(q);

  return (aHa + bHb) * hstep;
}

/*
inline size_t sq2d_idx(const size_t nxgrid, const size_t nygrid, size_t x, size_t y) { return (y * nxgrid) + x; }

void set_hamiltonian_sq2d(gsl_matrix *H, const gsl_vector *V, const double planck, const double mass, const double hstep, const size_t nxgrid, const size_t nygrid)
{
  ASSERT_SQUARE_SIZE(H, nxgrid * nygrid, "set_hamiltonian_2d_square: H does not match nxgrid and nygrid");
  ASSERT_SIZE1(H, V->size, "set_hamiltonian_2d_square: dim(H) != dim(V)");

  gsl_matrix_set_all(H, 0.0);

  const double pfact = -0.5 * planck * planck / mass;
  const double hstep2 = 1.0 / (hstep * hstep);

  for (int i = 1; i <= (nxgrid - 2); i++) {
    for (int j = 1; j <= (nygrid - 2); j++) {
      if (i > 1) { gsl_matrix_set(H, 
    }
  }
}
*/

timeevol_halves *timeevol_halves_alloc(const size_t N)
{
  timeevol_halves *U = malloc(sizeof(timeevol_halves));
  U->A = gsl_matrix_complex_calloc(N, N);
  U->B = gsl_matrix_complex_calloc(N, N);
  U->halfH = gsl_matrix_complex_calloc(N, N);
  U->ALU = gsl_matrix_complex_calloc(N, N);
  U->ALUp = gsl_permutation_alloc(N);
  return U;
}

void timeevol_halves_free(timeevol_halves *U)
{
  gsl_permutation_free(U->ALUp);
  gsl_matrix_complex_free(U->ALU);
  gsl_matrix_complex_free(U->halfH);
  gsl_matrix_complex_free(U->B);
  gsl_matrix_complex_free(U->A);
  free(U);
}

void set_timeevol_halves(timeevol_halves *U,
		     const gsl_matrix *H0, const gsl_matrix *H1,
		     const double planck, const double tstep, FILE *fdebug)
{
  const size_t N = H0->size1;

  ASSERT_SQUARE(H0, "timeevol_halves: H0 not square");
  ASSERT_SQUARE_SIZE(H1, N, "timeevol_halves: H1 bad size");
  ASSERT_SQUARE_SIZE(U->halfH, N, "timeevol_halves: U->halfH bad size");
  ASSERT_SQUARE_SIZE(U->A, N, "timeevol_halves: U->A bad size");
  ASSERT_SQUARE_SIZE(U->B, N, "timeevol_halves: U->B bad size");

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      gsl_complex *halfHij = gsl_matrix_complex_ptr(U->halfH, i, j);
      GSL_SET_COMPLEX(halfHij, 0.25 * (gsl_matrix_get(H0, i, j) + gsl_matrix_get(H1, i, j)), 0.0);
    }
  }

  if (fdebug) {
    fprintf(fdebug, "halfH = \n");
    fwrite_matrix_complex(fdebug, U->halfH);
  }
  
  GSL_SET_COMPLEX(&(U->ihdt), 0.0, planck / tstep);

  if (fdebug) {
    fprintf(fdebug, "ihΔt = %0.6f+%0.6fi\n", GSL_REAL(U->ihdt), GSL_IMAG(U->ihdt));
  }
  
  gsl_matrix_complex_set_identity(U->A);
  gsl_matrix_complex_scale(U->A, U->ihdt);
  gsl_matrix_complex_sub(U->A, U->halfH);

  if (fdebug) {
    fprintf(fdebug, "A = \n");
    fwrite_matrix_complex(fdebug, U->A);
  }

  gsl_matrix_complex_set_identity(U->B);
  gsl_matrix_complex_scale(U->B, U->ihdt);
  gsl_matrix_complex_add(U->B, U->halfH);

  if (fdebug) {
    fprintf(fdebug, "B = \n");
    fwrite_matrix_complex(fdebug, U->B);
  }

  int Asgn;
  gsl_matrix_complex_memcpy(U->ALU, U->A);
  gsl_linalg_complex_LU_decomp(U->ALU, U->ALUp, &Asgn);
}

void set_timeevol(gsl_matrix_complex *Uout, const gsl_matrix *H0, const gsl_matrix *H1, 
	        const double planck, const double hstep, const double tstep, FILE *fdebug)
{
  const int N = H0->size1;
  
  timeevol_halves *U = timeevol_halves_alloc(N);
  
  set_timeevol_halves(U, H0, H1, planck, tstep, fdebug);

  gsl_matrix_complex *Ainv = gsl_matrix_complex_alloc(N, N);
  gsl_linalg_complex_LU_invert(U->ALU, U->ALUp, Ainv);
  
  gsl_complex one, zero;
  GSL_SET_COMPLEX(&one, 1.0, 0.0);
  GSL_SET_COMPLEX(&zero, 0.0, 0.0);
  gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, one, Ainv, U->B, zero, Uout);
  
  if (fdebug) {
    fprintf(fdebug, "Ainv = \n");
    fwrite_matrix_complex(fdebug, Ainv);
  } 
  
  timeevol_halves_free(U);
  gsl_matrix_complex_free(Ainv);
}

void timeevol_state(gsl_vector_complex *psinew, const timeevol_halves *U, const gsl_vector_complex *psiold)
{
  gsl_complex one, zero;
  GSL_SET_COMPLEX(&one, 1.0, 0.0);
  GSL_SET_COMPLEX(&zero, 0.0, 0.0);

  gsl_blas_zgemv(CblasNoTrans, one, U->B, psiold, zero, psinew);    
  gsl_linalg_complex_LU_svx(U->ALU, U->ALUp, psinew);
}
  

void eigen_solve_alloc(const gsl_matrix *Hin, gsl_vector **eval, gsl_matrix **evec)
{
  const size_t STATESIZE = Hin->size1;
  if (Hin->size2 != STATESIZE) {
    fprintf(stderr, "eigen_solve_alloc: Hin not square");
    exit(1);
  }

  gsl_matrix *H = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_matrix_memcpy(H, Hin);
  
  *eval = gsl_vector_alloc(STATESIZE);
  *evec = gsl_matrix_alloc(STATESIZE, STATESIZE);

  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(STATESIZE);

  gsl_eigen_symmv(H, *eval, *evec, w);

  gsl_eigen_symmv_free(w);
  gsl_matrix_free(H);
  
  gsl_eigen_symmv_sort(*eval, *evec, GSL_EIGEN_SORT_VAL_ASC);
}

void eigen_norm_state_alloc(const gsl_matrix *evec, const double hstep, int state, gsl_vector_complex **psi_state)
{
  const int STATESIZE = evec->size1;
  if (evec->size2 != STATESIZE) {
    fprintf(stderr, "eigen_norm_state_alloc: evec not square");
    exit(1);
  }
  *psi_state = gsl_vector_complex_alloc(STATESIZE);
  double psi_norm = 0.0;
  for (int j = 0; j < STATESIZE; j++) {
    gsl_complex ej = gsl_complex_rect(gsl_matrix_get(evec, j, state), 0.0);
    gsl_vector_complex_set(*psi_state, j, ej);
    psi_norm += gsl_complex_abs2(ej) * hstep;
  }

  double sign = (gsl_matrix_get(evec, 1, state) > 0) ? 1.0 : (-1.0);
  gsl_vector_complex_scale(*psi_state, gsl_complex_rect(sign / sqrt(psi_norm), 0.0));
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
