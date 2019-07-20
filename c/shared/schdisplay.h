#ifndef _SCHDISPLAY_H
#define _SCHDISPLAY_H 1

void downsample_vector_complex_alloc(gsl_vector_complex **vv, const gsl_vector_complex *v, size_t n);

void terminal_graph_abs(const gsl_vector_complex *psi, const int, const double);
void terminal_graph_abs2(const gsl_vector_complex *psi, const int, const double);
void terminal_graph_phase(const gsl_vector_complex *psi, const int);

void terminal_graph_raw(const gsl_vector *v, double ymin, double ymax, 
			const unsigned int nlines, char filled);

#endif /* defined(_SCHDISPLAY_H) */
