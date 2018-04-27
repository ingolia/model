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

#include "params.h"
#include "potential.h"
#include "schdebug.h"
#include "schdisplay.h"
#include "schutil.h"
#include "schutil_1d.h"
#include "timeevol.h"

#define PLANCK 4.0
#define MASS   1.0
#define LENGTH (2.0 * M_PI)
#define V0MAX 64.0

#define STATESIZE 36
#define HSTEP (LENGTH / ((double) STATESIZE))
#define TSTEP (1.0/4096.0)

#define NSTATES 8

void test_6pts(const params *params);
void test_ramp(const params *params);
void evolve(const params *params, double tfinal);
void write_psi_complex(const char *fmt, const gsl_vector_complex *psi, int t);

int main(void)
{
  params params = { STATESIZE, PLANCK, TSTEP, HSTEP };
  
  // test_6pts(&params);
  // test_ramp(&params);
  evolve(&params, 40.0);
}

void test_6pts(const params *params)
{
  gsl_vector **Vs = calloc(64, sizeof(gsl_vector *));

  double Vpts[6];
  gsl_vector_complex **psis;
  double *Es;
  
  for (int i = 0; i < 64; i++) {
    Vpts[0] = (i & 0x01) ? V0MAX : 0.0;
    Vpts[1] = (i & 0x02) ? V0MAX : 0.0;
    Vpts[2] = (i & 0x04) ? V0MAX : 0.0;
    Vpts[3] = (i & 0x08) ? V0MAX : 0.0;
    Vpts[4] = (i & 0x10) ? V0MAX : 0.0;
    Vpts[5] = (i & 0x20) ? V0MAX : 0.0;

    Vs[i] = gsl_vector_calloc(params->statesize);
    potential_sin_6pt(Vs[i], Vpts);
    potential_test_stationary(params, Vs[i], MASS, NSTATES, &psis, &Es);
  }
}

#define VSTEP 0.5
#define RAMP_BASE "ring_six/ramp"

void test_ramp(const params *params)
{
  FILE *fE = fopenf("w", "%s-E.csv", RAMP_BASE);
  FILE *fpsi = fopenf("w", "%s-psi.csv", RAMP_BASE);

  gsl_vector *V = gsl_vector_calloc(params->statesize);

  double Vpts[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  gsl_vector_complex **psis;
  double *Es;

  for (double V0 = 0; V0 <= V0MAX; V0 += VSTEP) {
    Vpts[1] = V0;
    potential_sin_6pt(V, Vpts);
    potential_test_stationary(params, V, MASS, NSTATES, &psis, &Es);

    // Write energy eigenstates
    fprintf(fE, "%f", V0);
    for (int i = 0; i < NSTATES; i++) {
      fprintf(fE, ",%f", Es[i]);
    }
    fprintf(fE, "\n");

    // Write state vectors
    fprintf(fpsi, "%f", V0);
    for (int i = 0; i < NSTATES; i++) {
      for (int j = 0; j < psis[i]->size; j++) {
	fprintf(fpsi, ",%f", gsl_complex_abs(gsl_vector_complex_get(psis[i], j)));
      }
    }
    fprintf(fpsi, "\n");

    for (int i = 0; i < NSTATES; i++) {
      gsl_vector_complex_free(psis[i]);
    }
    free(psis);

    free(Es);
  }

  gsl_vector_free(V);

  fclose(fE);
  fclose(fpsi);
}

#define WRITEEVERY 128

void potential_t(gsl_vector *V, const double t) {
  double Vpts[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  const double Vmax = 6.0;

  Vpts[2] = Vmax * potential_asdr( 4.0,  6.0, 10.0, 12.0, t);
  Vpts[3] = Vmax * potential_asdr(15.0, 17.0, 18.0, 20.0, t);
  if (0) {
    Vpts[4] = Vmax * potential_asdr(27.0, 28.0, 28.0, 29.0, t);
    Vpts[5] = Vmax * potential_asdr(28.0, 29.0, 29.0, 30.0, t);
    Vpts[0] = Vmax * potential_asdr(29.0, 30.0, 30.0, 31.0, t);
    Vpts[1] = Vmax * potential_asdr(30.0, 31.0, 31.0, 32.0, t);
  }

  potential_sin_6pt(V, Vpts);
}
 
void evolve(const params *params, double tfinal)
{
  gsl_vector *V = gsl_vector_calloc(params->statesize);
  
  gsl_matrix *Hprev = gsl_matrix_alloc(params->statesize, params->statesize);
  gsl_matrix *H = gsl_matrix_alloc(params->statesize, params->statesize);

  gsl_vector_complex *psi = gsl_vector_complex_alloc(params->statesize);
  gsl_vector_complex *psinext = gsl_vector_complex_alloc(params->statesize);

  set_hamiltonian_spinor(Hprev, params, V, MASS);

  gsl_vector *eval;
  gsl_matrix *evec;
  gsl_vector_complex *psi_0, *psi_1;
  
  eigen_solve_alloc(Hprev, &eval, &evec);
  eigen_norm_state_alloc(evec, params->hstep, 0, &psi_0);
  eigen_norm_state_alloc(evec, params->hstep, 1, &psi_1);

  gsl_vector_free(eval);
  gsl_matrix_free(evec);

  gsl_vector_complex_scale(psi_0, gsl_complex_rect(M_SQRT1_2, 0.0));
  gsl_vector_complex_memcpy(psi, psi_0);
  
  gsl_vector_complex_scale(psi_1, gsl_complex_rect(0.0, M_SQRT1_2));
  gsl_vector_complex_add(psi, psi_1);

  gsl_vector_complex_free(psi_0);
  gsl_vector_complex_free(psi_1);

  timeevol *U = timeevol_alloc(params);
  
  for (int tstep = 0; (tstep * params->tstep) <= tfinal; tstep++) {
    const double t = tstep * params->tstep;

    potential_t(V, t);
    set_hamiltonian_spinor(H, params, V, MASS);

    if (tstep % WRITEEVERY == 0) {
      double E = get_energy(H, params, psi);
      
      printf("\033[2J\033[H");
      printf("t = %0.2f (tstep %6d), E = %0.2f\n", t, tstep, E);
      terminal_graph_abs2(psi, 24, 0.5);
      puts("");
      terminal_graph_phase(psi, 8);
      puts("");
      terminal_graph_raw(V, 0.0, 20.0, 10, '#');

      write_psi_complex("ring_six/evolve/psi-%08d.csv", psi, tstep);
    }

    timeevol_set_real_average(U, Hprev, H, params, NULL);
    timeevol_state(psinext, U, psi);

    gsl_vector_complex_memcpy(psi, psinext);
    gsl_matrix_memcpy(Hprev, H);
  }

  timeevol_free(U);

  gsl_vector_complex_free(psi);
  gsl_vector_complex_free(psinext);
}

void write_psi_complex(const char *fmt, const gsl_vector_complex *psi, int t)
{
  FILE *f = fopenf("w", fmt, t);

  for (int i = 0; i < psi->size; i++) {
    gsl_complex psi_i = gsl_vector_complex_get(psi, i);
    fprintf(f, "%+0.6f%+0.6fi\n", GSL_REAL(psi_i), GSL_IMAG(psi_i));
  }

  fclose(f);
}
