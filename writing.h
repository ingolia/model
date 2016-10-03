#ifndef _WRITING_H
#define _WRITING_H

void fwrite_vector_complex_thorough(FILE *f, const gsl_vector_complex *V);
void fwrite_matrix_complex(FILE *f, const gsl_matrix_complex *M);

void terminal_graph_abs2(const gsl_vector_complex *psi, const int, const double);

#endif /* defined(_WRITING_H) */
