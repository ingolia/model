#define _GNU_SOURCE
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_complex_double.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_complex_double.h>

#include "schdebug.h"
#include "schdisplay.h"
#include "schutil.h"
#include "schutil_1d.h"

#include "potential.h"

void potential_sin_6pt(gsl_vector *V, double Vpts[6])
{
  const int sixth = V->size / 6;

  assert(sixth * 6 == V->size); // Require equal sixths!

  gsl_vector_set_all(V, 0.0);
  for (int pt = 0; pt < 6; pt++) {
    for (int pti = 0; pti <= 2 * sixth; pti++) {
      const double th = M_PI * ((double) pti) / ((double) sixth);
      const int i0 = (pt * sixth) + pti;
      const int i = (i0 >= V->size) ? (i0 - V->size) : i0;
      assert(i >= 0 && i < V->size);
      *gsl_vector_ptr(V, i) += Vpts[pt] * (1.0 - cos(th));
    }
  }
}

double potential_cyclic(const double tcycle, 
			unsigned int nphases, 
			unsigned int phase, 
			const double t)
{
  const double tincycle = t - tcycle * ((int) (t / tcycle));
  assert(tincycle >= 0.0 && tincycle <= tcycle);
  const double tphase = tincycle * ((double) nphases) / tcycle;
  assert(tphase >= 0.0 && tphase <= ((double) nphases));
  const double relphase = tphase - (double) phase;

  if ((relphase < 0.0) || (relphase > 2.0)) {
    return 0.0;
  } else {
    return sin(relphase * M_PI / 2.0);
  }
}

double potential_asdr(const double ta, const double ts, 
		      const double td, const double tr, 
		      const double t)
{
  if ((t < ta) || (t > tr)) {
    return 0.0;
  } else if (t < ts) {
    return (t - ta) / (ts - ta);
  } else if (t < td) {
    return 1.0;
  } else {
    return (tr - t) / (tr - td);
  }
}
void potential_control(const params *params,
		       const gsl_vector *V,
		       const double mass)
{
  const size_t statesize = params->statesize;
  gsl_matrix *H = gsl_matrix_alloc(statesize, statesize);

  set_hamiltonian_circular(H, params, V, mass);
  
  gsl_vector *eval_V;
  gsl_matrix *evec_V;

  eigen_solve_alloc(H, &eval_V, &evec_V);

  gsl_vector *V0 = gsl_vector_calloc(statesize);

  set_hamiltonian_circular(H, params, V0, mass);

  gsl_vector *eval_0;
  gsl_matrix *evec_0;

  eigen_solve_alloc(H, &eval_0, &evec_0);

  FILE *f = fopenf("w", "ring_six/control-V.txt");

  for (int i = 0; i < statesize; i++) {
    gsl_vector_view evec_Vi = gsl_matrix_column(evec_V, i);

    fprintf(f, "%d\t%0.2f", i, gsl_vector_get(eval_V, i));
    for (int j = 0; j < 5; j++) {
      gsl_vector_view evec_0j = gsl_matrix_column(evec_0, j);

      double x;
      gsl_blas_ddot(&evec_Vi.vector, &evec_0j.vector, &x);
      fprintf(f, "\t%+0.3f", x);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  f = fopenf("w", "ring_six/control-0.txt");

  for (int i = 0; i < statesize; i++) {
    gsl_vector_view evec_0i = gsl_matrix_column(evec_0, i);

    fprintf(f, "%d\t%0.2f", i, gsl_vector_get(eval_0, i));
    for (int j = 0; j < 5; j++) {
      gsl_vector_view evec_Vj = gsl_matrix_column(evec_V, j);

      double x;
      gsl_blas_ddot(&evec_0i.vector, &evec_Vj.vector, &x);
      fprintf(f, "\t%+0.3f", x);
    }
    fprintf(f, "\n");
  }

  fclose(f);
}

void potential_test_stationary(const params *params,
			       const gsl_vector *V,
			       const double mass,
			       unsigned int nstates,
			       gsl_vector_complex ***psis,
			       double **Es)
{
  const size_t statesize = params->statesize;
  gsl_matrix *H = gsl_matrix_alloc(statesize, statesize);

  set_hamiltonian_circular(H, params, V, mass);
  
  gsl_vector *eval;
  gsl_matrix *evec;

  *Es = calloc(nstates, sizeof(double));
  *psis = calloc(nstates, sizeof(gsl_vector_complex *));
  
  eigen_solve_alloc(H, &eval, &evec);

  for (int i = 0; (i < nstates); i++) {
    eigen_norm_state_alloc(evec, params->hstep, i, &((*psis)[i]));

    (*Es)[i] = get_energy(H, params, (*psis)[i]);

    printf("\033[2J\033[H");
    printf("i = %d\nE = %0.2f\n", i, (*Es)[i]);
    terminal_graph_abs2((*psis)[i], 12, 0.5);
    puts("");
    terminal_graph_phase((*psis)[i], 8);

    puts("");
    terminal_graph_raw(V, gsl_vector_min(V), gsl_vector_max(V), 16, '%');

    sleep(3);
  }

  gsl_matrix_free(H);
}
