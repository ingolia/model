#ifndef _GRID2D_H
#define _GRID2D_H 1

/* Undirected edges in adjacency graph */
typedef struct {
  size_t v1;
  size_t v2;
} edge2d;

typedef struct {
  size_t npts;     /* Total number of points in grid */
  long   eta0;     /* Eta coordinate of the first row */
  size_t neta;     /* Number of rows indexed by eta */

  long   *chi0s;   /* First chi coordinate "column" in a row */
  size_t *nchis;   /* Number of chi coordinate columns in a row */
  size_t *chi0idx; /* Index of point at first chi coordinate in a row */
  
  long   *idxeta;  /* Eta coordinate of a point (length = npts) */
  long   *idxchi;  /* Chi coordinate of a point (length = npts) */

  size_t  nedges;  /* Number of edges (size of array) */
  edge2d *edges;   /* Edges in adjacency graph */

  size_t  nbndry;  /* Number of points in the boundary */
  size_t *bndry;
} grid2d;

void grid2d_free(grid2d *grid);

grid2d *grid2d_new_rectangle(const size_t nchi, const size_t neta);
void grid2d_set_laplacian(gsl_matrix *L, const grid2d *grid);

inline size_t grid2d_index(const grid2d *g, long chi, long eta)
{
  const long deta = eta - g->eta0;
  const long dchi = chi - g->chi0s[(size_t) deta];
  return g->chi0idx[(size_t) deta] + dchi;
}

void validate_grid2d(const grid2d *grid);

#endif
