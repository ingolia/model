#ifndef _WRITING_H
#define _WRITING_H

void fwrite_vector_complex_thorough(FILE *f, const gsl_vector_complex *V);
void fwrite_matrix_complex(FILE *f, const gsl_matrix_complex *M);

#endif /* defined(_WRITING_H) */
