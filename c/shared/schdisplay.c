#include <math.h>
#include <stdio.h>

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_complex_double.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_complex_double.h>

#include "schdisplay.h"

#define MIN_ABS2 1e-6

void terminal_graph_abs(const gsl_vector_complex *psi, const int nlines, const double ymax)
{
  gsl_vector *h = gsl_vector_alloc(psi->size);
  for (int i = 0; i < psi->size; i++) {
    gsl_vector_set(h, i, gsl_complex_abs(gsl_vector_complex_get(psi, i)));
  }

  terminal_graph_raw(h, 0.0, ymax, nlines, '#');
  gsl_vector_free(h);
}

void terminal_graph_abs2(const gsl_vector_complex *psi, const int nlines, const double ymax)
{
  gsl_vector *h = gsl_vector_alloc(psi->size);
  for (int i = 0; i < psi->size; i++) {
    gsl_vector_set(h, i, gsl_complex_abs2(gsl_vector_complex_get(psi, i)));
  }

  terminal_graph_raw(h, 0.0, ymax, nlines, '*');
  gsl_vector_free(h);
}

void terminal_graph_phase(const gsl_vector_complex *psi, const int nlines)
{
  gsl_vector *h = gsl_vector_alloc(psi->size);
  gsl_vector *g = gsl_vector_alloc(psi->size);
  for (int i = 0; i < psi->size; i++) {
    gsl_vector_set(h, i, gsl_complex_arg(gsl_vector_complex_get(psi, i)) + M_PI);
    gsl_vector_set(g, i, gsl_complex_abs2(gsl_vector_complex_get(psi, i)));
  }

  double lineheight = (2.0 * M_PI) / ((double) nlines);
  for (int j = nlines; j > 0; j--) {
    double linemin = (((double) j) - 1.0) * lineheight;
    double linemax = (((double) j)      ) * lineheight;
    printf(" %6.2f |", linemin);
    for (int i = 0; i < psi->size; i++) {
      if ((gsl_vector_get(g, i) > MIN_ABS2) &&
	(gsl_vector_get(h, i) > linemin && gsl_vector_get(h, i) <= linemax)) {
        putchar('-');
      } else {
        putchar(' ');
      }
    }
    puts("|");
  }
}

void terminal_graph_raw(const gsl_vector *v, double ymin, double ymax, 
			const unsigned int nlines, char filled)
{
  const double lineheight = (ymax - ymin) / ((double) nlines);

  for (int j = nlines; j > 0; j--) {
    double hthresh = ymin + (((double) j) - 0.5) * lineheight;
    printf("% 7.2f |", hthresh);
    for (int i = 0; i < v->size; i++) {
      if (gsl_vector_get(v, i) > hthresh) {
        putchar(filled);
      } else {
        putchar(' ');
      }
    }
    puts("|");    
  }
}

void downsample_vector_complex_alloc(gsl_vector_complex **vv, const gsl_vector_complex *v, size_t n)
{
  size_t vvsize = v->size / n;

  *vv = gsl_vector_complex_alloc(vvsize);

  const double ninv = 1.0 / ((double) n);
  for (size_t i = 0; i < vvsize; i++) {
    gsl_complex avg = gsl_vector_complex_get(v, i*n);
    for (size_t j = 1; j < n; j++) {
      avg = gsl_complex_add(avg, gsl_vector_complex_get(v, (i*n) + j));
    }
    gsl_vector_complex_set(*vv, i, gsl_complex_mul_real(avg, ninv));
  }
}
