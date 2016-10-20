#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_matrix.h>

#include "grid2d.h"

typedef struct edge2d_list_struct {
  edge2d edge;
  struct edge2d_list_struct *next;
} edge2d_list;

void add_edge(edge2d_list **lptr, edge2d e);
void free_list(edge2d_list *l);
void list_to_array(const edge2d_list *edgelist, size_t *nedges, edge2d **edges);

void list_to_array(const edge2d_list *edgelist, size_t *nedges, edge2d **edges)
{
  (*nedges) = 0;
  const edge2d_list *e = edgelist;
  while (e != NULL) {
    (*nedges)++;
    e = e->next;
  }

  (*edges) = calloc((*nedges), sizeof(edge2d));
  size_t i;
  for (i = 0, e = edgelist; i < (*nedges) && e != NULL; i++, e = e->next) {
    (*edges)[i] = e->edge;
  }
}

void add_edge(edge2d_list **lptr, edge2d e)
{
  edge2d_list *lnew = calloc(1, sizeof(edge2d_list));
  lnew->edge = e;
  lnew->next = (*lptr);
  (*lptr) = lnew;
}

void free_list(edge2d_list *l)
{
  edge2d_list *next;

  while (l != NULL) {
    next = l->next;
    free(l);
    l = next;
  }
}

void grid2d_free(grid2d *grid)
{
  free(grid->chi0s);
  free(grid->nchis);
  free(grid->chi0idx);
  
  free(grid->idxeta);
  free(grid->idxchi);

  free(grid->edges);

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

  edge2d_list *edgelist = NULL;

  for (int chi = 0; chi < nchi; chi++) {
    for (int eta = 0; eta < neta; eta++) {
      edge2d e;

      if (eta + 1 < neta) { 
	e.v1 = grid2d_index(grid, chi, eta);
	e.v2 = grid2d_index(grid, chi, eta + 1);
	add_edge(&edgelist, e);
      }

      if (chi + 1 < nchi) {
	e.v1 = grid2d_index(grid, chi, eta);
	e.v2 = grid2d_index(grid, chi + 1, eta);
	add_edge(&edgelist, e);
      }
    }
  }

  list_to_array(edgelist, &(grid->nedges), &(grid->edges));
  free_list(edgelist);

  return grid;
}

void grid2d_set_laplacian(gsl_matrix *L, const grid2d *grid)
{
  if ((L->size1 != grid->npts) || (L->size2 != grid->npts)) {
    fprintf(stderr, "grid2d_set_laplacian: matrix size (%lu, %lu) does not match grid size %lu\n", 
	    L->size1, L->size2, grid->npts);
    exit(1);
  }

  gsl_matrix_set_all(L, 0.0);

  for (size_t i = 0; i < grid->nedges; i++) {
    const edge2d e = grid->edges[i];
    (*gsl_matrix_ptr(L, e.v1, e.v1)) += 1.0;
    (*gsl_matrix_ptr(L, e.v2, e.v2)) += 1.0;
    (*gsl_matrix_ptr(L, e.v1, e.v2)) -= 1.0;
    (*gsl_matrix_ptr(L, e.v2, e.v1)) -= 1.0;
  }

  
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
