#include <math.h>
#include <stdio.h>

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_complex_double.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_complex_double.h>

#define MIN_ABS2 1e-6

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

void terminal_graph_abs(const gsl_vector_complex *psi, const int nlines, const double ymax)
{
  gsl_vector *h = gsl_vector_alloc(psi->size);
  for (int i = 0; i < psi->size; i++) {
    gsl_vector_set(h, i, gsl_complex_abs(gsl_vector_complex_get(psi, i)) / ymax);
  }

  terminal_graph_raw(h, nlines, '#');
  gsl_vector_free(h);
}

void terminal_graph_abs2(const gsl_vector_complex *psi, const int nlines, const double ymax)
{
  gsl_vector *h = gsl_vector_alloc(psi->size);
  for (int i = 0; i < psi->size; i++) {
    gsl_vector_set(h, i, gsl_complex_abs2(gsl_vector_complex_get(psi, i)) / ymax);
  }

  terminal_graph_raw(h, nlines, '*');
  gsl_vector_free(h);
}

void terminal_graph_raw(const gsl_vector *v, const unsigned int nlines, char filled) {
  const double lineheight = 1.0 / ((double) nlines);

  for (int j = nlines; j > 0; j--) {
    double hthresh = (((double) j) - 0.5) * lineheight;
    printf("%0.4f |", hthresh);
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
    printf("%0.4f |", linemin);
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
