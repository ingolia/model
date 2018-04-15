#ifndef _WRITING_H
#define _WRITING_H

void fwrite_vector(FILE *f, const gsl_vector *V);
void fwrite_vector_complex_thorough(FILE *f, const gsl_vector_complex *V);

void fwrite_vector_complex_abs2(FILE *f, const gsl_vector_complex *psi);
void fwrite_vector_complex_abs(FILE *f, const gsl_vector_complex *psi);
void fwrite_vector_complex_arg(FILE *f, const gsl_vector_complex *psi);

void fwrite_matrix_complex(FILE *f, const gsl_matrix_complex *M);
void fwrite_matrix(FILE *f, const gsl_matrix *M);

void downsample_vector_complex_alloc(gsl_vector_complex **vv, const gsl_vector_complex *v, unsigned int n);

void terminal_graph_abs(const gsl_vector_complex *psi, const int, const double);
void terminal_graph_abs2(const gsl_vector_complex *psi, const int, const double);
void terminal_graph_phase(const gsl_vector_complex *psi, const int);
void terminal_graph_raw(const gsl_vector *v, const unsigned int nlines, char filled);

#endif /* defined(_WRITING_H) */
