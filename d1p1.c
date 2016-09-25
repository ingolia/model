#include <math.h>
#include <stdio.h>
#include <unistd.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#define NPTS 126
#define STATESIZE (NPTS + 2)

#define MASS 1.0
#define HSTEP (1.0/64.0)

void solve_potential(const char *prefix, const gsl_vector *V);
void eig_table(gsl_matrix **Psis, gsl_vector **Es, const gsl_vector *eval, const gsl_matrix *evec);
void write_potential(const char *prefix, const gsl_vector *V);
void write_hermitian(const char *prefix, const gsl_matrix *H);
void write_energies(const char *prefix, const gsl_matrix *Psis, const gsl_vector *Es);
void write_psi(const char *prefix, const gsl_matrix *Psis, const gsl_vector *Es);
void set_H(gsl_matrix *H, const gsl_vector *V);

int main(void)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  solve_potential("data/none", V);

  for (int j = NPTS/2; j <= NPTS; j++) {
    gsl_vector_set(V, j, 50.0);
  }

  solve_potential("data/half", V);
}

void solve_potential(const char *prefix, const gsl_vector *V)
{
  write_potential(prefix, V);

  gsl_matrix *H = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_H(H, V);

  write_hermitian(prefix, H);

  gsl_vector *eval = gsl_vector_alloc(STATESIZE);
  gsl_matrix *evec = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(STATESIZE);

  gsl_eigen_symmv(H, eval, evec, w);

  gsl_eigen_symmv_free(w);

  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_ASC);

  gsl_matrix *Psis;
  gsl_vector *Es;

  eig_table(&Psis, &Es, eval, evec);
  write_energies(prefix, Psis, Es);
  write_psi(prefix, Psis, Es);
}

void set_H(gsl_matrix *H, const gsl_vector *V)
{
  gsl_matrix_set_all(H, 0.0);

  for (int j = 1; j <= NPTS; j++) {
    const double pfact = -0.5 / MASS;
    const double hstep2 = 1.0 / (HSTEP * HSTEP);

    if (j > 1)    { gsl_matrix_set(H, j, j-1, pfact * hstep2); }
    if (j < NPTS) { gsl_matrix_set(H, j, j+1, pfact * hstep2); }

    gsl_matrix_set(H, j, j, -2.0 * pfact * hstep2 + gsl_vector_get(V, j));
  }
}

void eig_table(gsl_matrix **Psis, gsl_vector **Es, const gsl_vector *eval, const gsl_matrix *evec)
{
  (*Psis) = gsl_matrix_alloc(STATESIZE, STATESIZE);
  (*Es) = gsl_vector_alloc(STATESIZE);

  gsl_vector *psi = gsl_vector_alloc(STATESIZE);

  for (int st = 0; st < STATESIZE; st++) {
    gsl_matrix_get_col(psi, evec, st);
    gsl_matrix_set_col(*Psis, st, psi);
    gsl_vector_set(*Es, st, gsl_vector_get(eval, st));
    printf("E_%d\t%0.4f\n", st, gsl_vector_get(eval, st));
  }
  gsl_vector_free(psi);
}

void write_potential(const char *prefix, const gsl_vector *V)
{
  char *stname;
  asprintf(&stname, "%s_V.txt", prefix);
  FILE *fpot;
  if ((fpot = fopen(stname, "w")) == NULL) {
    fprintf(stderr, "Cannot open output file \"%s\"\n", stname);
    exit(1);
  }
  free(stname);

  for (int j = 1; j <= NPTS; j++) {
    fprintf(fpot, "%d\t%0.6f\n", j, gsl_vector_get(V, j));
  }

  fclose(fpot);
}

void write_hermitian(const char *prefix, const gsl_matrix *H)
{
  FILE *fout;

  char *stname;
  asprintf(&stname, "%s_H.txt", prefix);
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
  asprintf(&stname, "%s_E.txt", prefix);
  if ((fout = fopen(stname, "w")) == NULL) {
    fprintf(stderr, "Cannot open output file \"%s\"\n", stname);
    exit(1);
  }
  free(stname);

  gsl_vector *psi = gsl_vector_alloc(STATESIZE);

  fprintf(fout, "phi_n\tE_n\n");
  for (int st = 0; st + 1 < Es->size; st++) {
    fprintf(fout, "%d\t%0.6f", st, gsl_vector_get(Es, st));      
    fprintf(fout, "\n");
  }

  gsl_vector_free(psi);
  fclose(fout);
}

void write_psi(const char *prefix, const gsl_matrix *Psis, const gsl_vector *Es)
{
  FILE *fout;

  char *stname;
  asprintf(&stname, "%s_psi.txt", prefix);
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
  
  for (int j = 0; j < STATESIZE; j++) {
    fprintf(fout, "%d", j);

    for (int st = 0; st < Psis->size2; st++) {
      double psi_st_j = gsl_matrix_get(Psis, j, st);
      fprintf(fout, "\t%0.6f", psi_st_j);
    }

    fprintf(fout, "\n");
  }

  
  fclose(fout);
}

