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
#include "schutil.h"
#include "schutil_1d.h"

#define PLANCK 4.0
#define MASS   1.0
#define LENGTH (2.0 * M_PI)
#define V0MAX 8.0

#define STATESIZE 36
#define HSTEP (LENGTH / ((double) STATESIZE))
#define TSTEP (1.0/4096.0)

#define NSTATES 7

void test_6pts(const params *params);
void test_ramp(const params *params);

int main(void)
{
  params params = { STATESIZE, PLANCK, TSTEP, HSTEP };

  //  test_6pts(&params);
  test_ramp(&params);
}

void test_6pts(const params *params)
{
  gsl_vector **Vs = calloc(64, sizeof(gsl_vector *));

  double Vpts[6];
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
    potential_test_stationary(params, Vs[i], MASS, NSTATES, &Es);
  }
}

#define VSTEP 0.1
#define RAMP_FILE "ring_six/ramp.csv"

void test_ramp(const params *params)
{
  FILE *f = fopen(RAMP_FILE, "w");
  if (f == NULL) {
    fprintf(stderr, "test_ramp: could not open output file \"%s\"\n", RAMP_FILE);
    exit(1);
  }

  gsl_vector *V = gsl_vector_calloc(params->statesize);

  double Vpts[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  double *Es;

  for (double V0 = 0; V0 <= V0MAX; V0 += VSTEP) {
    Vpts[2] = V0;
    potential_sin_6pt(V, Vpts);
    potential_test_stationary(params, V, MASS, NSTATES, &Es);

    fprintf(f, "%f", V0);
    for (int i = 0; i < NSTATES; i++) {
      fprintf(f, ",%f", Es[i]);
    }
    fprintf(f, "\n");

    free(Es);
  }

  gsl_vector_free(V);

  fclose(f);
}
