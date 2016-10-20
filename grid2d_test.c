#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_matrix.h>

#include "grid2d.h"
#include "writing.h"

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

  g1 = grid2d_new_rectangle(3, 4);

  for (size_t i = 0; i < g1->npts; i++) {
    printf("%2lu -> (%ld, %ld)\n", 
	   i, g1->idxchi[i], g1->idxeta[i]);
  }

  gsl_matrix *L = gsl_matrix_calloc(g1->npts, g1->npts);
  grid2d_set_laplacian(L, g1);
  grid2d_free(g1);

  fwrite_matrix(stdout, L);

  gsl_matrix_free(L);
}
