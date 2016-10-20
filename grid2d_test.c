#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_vector.h>

#include "grid2d.h"
#include "schutil.h"
#include "writing.h"

#define PLANCK 1.0
#define MASS   1.0

#define NX 16
#define NY 16

#define HSTEP  (1.0 / ((double) NX))

#define BUFLEN 1024
static char buf[BUFLEN];


int main(void)
{
  grid2d *g1 = grid2d_new_rectangle(17, 23);
  grid2d *g2 = grid2d_new_rectangle(23, 17);
  grid2d *g3 = grid2d_new_rectangle(19, 19);

  validate_grid2d(g1);
  validate_grid2d(g2);
  validate_grid2d(g3);

  grid2d_free(g1);
  grid2d_free(g2);
  grid2d_free(g3);

  g1 = grid2d_new_rectangle(5, 4);

  for (size_t i = 0; i < g1->npts; i++) {
    printf("%2lu -> (%ld, %ld)\n", 
	   i, g1->idxchi[i], g1->idxeta[i]);
  }

  gsl_matrix *L = gsl_matrix_calloc(g1->npts, g1->npts);
  grid2d_set_laplacian(L, g1);

  fwrite_matrix(stdout, L);
  grid2d_free(g1);
  gsl_matrix_free(L);

  g1 = grid2d_new_rectangle(16, 16);

  gsl_matrix *H = gsl_matrix_calloc(g1->npts, g1->npts);
  gsl_vector *V = gsl_vector_calloc(g1->npts);
  
  set_hamiltonian_sq2d(H, V, PLANCK, MASS, HSTEP, g1);

  gsl_vector *eval;
  gsl_matrix *evec;
  gsl_vector_complex *psi;
  
  eigen_solve_alloc(H, &eval, &evec);
  
  FILE *fval = fopen("grid2ddata/eig-v0-energies.txt", "w");

  for (size_t j = 0; j < evec->size2; j++) {
    fprintf(fval, "%04lu\t%0.6f\n", j, gsl_vector_get(eval, j));

    snprintf(buf, BUFLEN, "grid2ddata/eig-v0-%04lu-abs2.txt", j);
    FILE *fvec = fopen(buf, "w");
    eigen_norm_state_alloc(evec, HSTEP, j, &psi);
    
    for (size_t eta = 0; eta < NY; eta++) {
      for (size_t chi = 0; chi < NX; chi++) {
	fprintf(fvec, "%0.6f%c", 
		gsl_complex_abs2(gsl_vector_complex_get(psi, grid2d_index(g1, chi, NY - (eta + 1)))),
		(chi == (NX - 1)) ? '\n' : '\t');
      }
    }

    gsl_vector_complex_free(psi);
    fclose(fvec);
  }
  fclose(fval);

  gsl_matrix_free(evec);
  gsl_vector_free(eval);
  gsl_matrix_free(H);
  gsl_vector_free(V);
  grid2d_free(g1);
}
