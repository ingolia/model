#ifndef _SCHDEBUG_H
#define _SCHDEBUG_H 1

#define ASSERT_SQUARE(m, msg) { if ((m->size1) != (m->size2)) { fprintf(stderr, "%s (%lu x %lu)", msg, m->size1, m->size2); exit(1); } }
#define ASSERT_SIZE1(m, sz, msg) { if ((m->size1 != sz)) { fprintf(stderr, "%s (%lu != %lu)", msg, m->size1, sz); exit(1); } }
#define ASSERT_SQUARE_SIZE(m, sz, msg) { if ((m->size1 != m->size2) || (m->size1 != sz)) { fprintf(stderr, "%s (%lu x %lu != %lu)", msg, m->size1, m->size2, sz); exit(1); } }

void write_potential(const char *prefix, const gsl_vector *V);
void write_hamiltonian(const char *prefix, const gsl_matrix *H);
void write_energies(const char *prefix, const gsl_matrix *Psis, const gsl_vector *Es);
void write_psi(const char *prefix, const gsl_matrix *Psis, const gsl_vector *Es);

void fwrite_vector(FILE *f, const gsl_vector *V);
void fwrite_vector_complex_thorough(FILE *f, const gsl_vector_complex *V);

void fwrite_vector_complex_abs2(FILE *f, const gsl_vector_complex *psi);
void fwrite_vector_complex_abs(FILE *f, const gsl_vector_complex *psi);
void fwrite_vector_complex_arg(FILE *f, const gsl_vector_complex *psi);

void fwrite_matrix_complex(FILE *f, const gsl_matrix_complex *M);
void fwrite_matrix(FILE *f, const gsl_matrix *M);

void check_symmetry(const gsl_matrix *M);
void check_unitarity(const gsl_matrix_complex *M, FILE *fdebug);

void check_deviation(const gsl_matrix *Z, size_t *imax, size_t *jmax, double *absmax, double *norm);

FILE *fopenf(const char *mode, const char *fmt, ...);

#endif /* !defined(_SCHDEBUG_H) */
