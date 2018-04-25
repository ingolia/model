#ifndef _TIMEEVOL_H
#define _TIMEEVOL_H 1

#include "params.h"

/* Time evolution matrices */
/* H_0, H_1 = Hamiltonians at t = t0, t0 + tstep
 * Make H = (H_0 + H_1) / 2
 * i \hbar d\Psi/dt = H \Psi
 * d\Psi/dt ~ (1 / tstep) (\Psi(t + tstep) - \Psi(t))
 * i \hbar (1 / tstep) (\Psi(t + tstep) - \Psi(t)) = 0.5 (H \Psi(t + tstep) + H \Psi(t))
 * i \hbar (1 / tstep) \Psi(t + tstep) - 0.5 H \Psi(t + tstep) 
 *   = i \hbar (1 / tstep) \Psi(t) + 0.5 H \Psi(t)
 * (i \hbar (1 / tstep) I - 0.5 H) \Psi(t + step)
 *   = (i \hbar (1 / tstep) I + 0.5 H) \Psi(t)
 * A = ((i \hbar / tstep) I - 0.5 H)
 * B = ((i \hbar / tstep) I + 0.5 H)
 * A Psi(t + tstep) = B Psi(t) OR Psi_next = A^{-1} B Psi_curr
 *   (A^{-1} B)^*
 *   = ((i c I - 0.5 H)^{-1} (i c I + 0.5 H))^*
 *   ...
 *   = (A^{-1} B)^{-1}
 * i.e., unitary
 */

typedef struct {
  gsl_matrix_complex *B;
  gsl_matrix_complex *ALU; /* LU decomp of A */
  gsl_permutation *ALUp;   
} timeevol;

timeevol *timeevol_alloc(const params *params);
void timeevol_free(timeevol *U);

void timeevol_set_real(timeevol *U,
		       const gsl_matrix *Hreal,
		       const params *params,
		       FILE *fdebug);

void timeevol_set_real_average(timeevol *U,
			       const gsl_matrix *Hreal1,
			       const gsl_matrix *Hreal2,
			       const params *params,
			       FILE *fdebug);

void timeevol_set(timeevol *U,
		  const gsl_matrix_complex *Havg,
		  const params *params,
		  FILE *fdebug);

void timeevol_state(gsl_vector_complex *, const timeevol *, const gsl_vector_complex *);

#endif
