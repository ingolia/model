#include <stdio.h>
#include <stdlib.h>

#include "grid2d.h"

void grid2d_free(grid2d *grid)
{
  free(grid->chi0s);
  free(grid->nchis);
  free(grid->chi0idx);
  
  free(grid->idxeta);
  free(grid->idxchi);

  free(grid);
}

grid2d *grid2d_new_rectangle(const size_t nchi, const size_t neta)
{
  grid2d *grid = calloc(1, sizeof(grid2d));
  grid->npts = nchi * neta;

  grid->eta0 = 0;
  grid->neta = neta;

  grid->chi0s = calloc(neta, sizeof(long));
  grid->nchis = calloc(neta, sizeof(size_t));
  grid->chi0idx = calloc(neta, sizeof(size_t));

  for (size_t deta = 0; deta < neta; deta++) {
    grid->chi0s[deta]   = 0;
    grid->nchis[deta]   = nchi;
    grid->chi0idx[deta] = deta * nchi;
  }

  grid->idxeta = calloc(grid->npts, sizeof(long));
  grid->idxchi = calloc(grid->npts, sizeof(long));
  for (size_t i = 0; i < grid->npts; i++) {
    grid->idxeta[i] = i / nchi;
    grid->idxchi[i] = i % nchi;
  }

  return grid;
}

void validate_grid2d(const grid2d *grid)
{
  for (size_t i = 0; i < grid->npts; i++) {
    long eta = grid->idxeta[i];
    long chi = grid->idxchi[i];

    long deta = eta - grid->eta0;
    if (deta < 0 || deta >= grid->neta) {
      fprintf(stderr, "%lu -> (chi = %ld, eta = %ld) and eta out of [%ld, %ld]\n",
	    i, chi, eta, grid->eta0, grid->eta0 + grid->neta - 1);
      exit(1);
    }

    long chi0 = grid->chi0s[(size_t) deta];
    long dchi = chi - chi0;
    if (dchi < 0 || dchi >= grid->nchis[(size_t) deta]) {
      fprintf(stderr, "%lu -> (chi = %ld, eta = %ld) and chi out of [%ld, %ld]\n",
	    i, chi, eta, chi0, chi0 + grid->nchis[(size_t) deta] - 1);
      exit(1);
    }

    size_t j = grid->chi0idx[(size_t) deta] + ((size_t) dchi);

    if (i != j) {
      fprintf(stderr, "%lu -> (chi = %ld, eta = %ld) -> %lu", i, chi, eta, j);
      fprintf(stderr, "  Δeta = %ld", deta);
      fprintf(stderr, "  chi0[Δeta] = %ld, Δchi = %ld", chi0, dchi);
      fprintf(stderr, "  index of chi0[Δeta] = %ld", grid->chi0idx[(size_t) deta]);
      fprintf(stderr, "  chi0idx[Δeta] + Δchi = %ld", j);
      exit(1);
    }

    size_t k = grid2d_index(grid, chi, eta);
    if (i != k) {
      fprintf(stderr, "%lu -> (%ld, %ld) and grid2d_index(%ld, %ld) -> %lu",
	    i, chi, eta, chi, eta, k);
      exit(1);
    }
  }
}
