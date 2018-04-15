#define _GNU_SOURCE
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

#include "schutil.h"
#include "writing.h"

#define NPTS 64
#define STATESIZE NPTS

#define PLANCK 4.0
#define MASS   1.0
#define V0MAX  5.0
#define HSTEP (8.0 / ((double) NPTS))
#define TSTEP (8.0/65536.0)

#define WRITEEVERY 512

#define NSTEPS 16

typedef struct {
  timeevol_halves *Usteady[NSTEPS];
  timeevol_halves *Uupfrom[NSTEPS];
  timeevol_halves *Udownto[NSTEPS];
} precomputed_timeevol;

void solve_stationary(void);
void evolve(gsl_vector *const[NSTEPS], const precomputed_timeevol *Uall);

precomputed_timeevol *precompute(gsl_vector *const Vs[NSTEPS]);

void v_sin(gsl_vector *V, double scale);

int main(void)
{
  solve_stationary();

  gsl_vector **Vs = calloc(NSTEPS, sizeof(gsl_vector *));
  for (int i = 0; i < NSTEPS; i++) {
    Vs[i] = gsl_vector_calloc(STATESIZE);
    v_sin(Vs[i], V0MAX * ((double) i) / ((double) (NSTEPS - 1)));
  }
  
  precomputed_timeevol *Uall = precompute(Vs);

  evolve(Vs, Uall);
}

void solve_stationary()
{
  char *prefix, *filename;
  asprintf(&prefix, "circdata/psi-h%0.3f-m%0.3f-n%03d-h%0.3f-t%0.3f",
	 PLANCK, MASS, NPTS, HSTEP, TSTEP);

  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian_circular(H0, V, PLANCK, MASS, HSTEP);

  gsl_vector *eval;
  gsl_matrix *evec;

  gsl_vector_complex **psis;
  psis = malloc(sizeof(gsl_vector_complex *) * STATESIZE);
  
  eigen_solve_alloc(H0, &eval, &evec);

  double *ts = calloc(STATESIZE, sizeof(double));
  double **targs = calloc(STATESIZE, sizeof(double *));
  for (int i = 0; i < STATESIZE; i++) {
    targs[i] = calloc(STATESIZE, sizeof(double));
  }
  
  asprintf(&filename, "%s-eigstates.txt", prefix);
  FILE *f = fopen(filename, "w");
  free(filename);

  fprintf(f, "# planck = %0.6f\n", PLANCK);
  fprintf(f, "# mass   = %0.6f\n", MASS);
  fprintf(f, "# npts   = %6d\n", NPTS);
  fprintf(f, "# hstep = %0.6f\n", HSTEP);
  fprintf(f, "# tstep  = %0.6f\n", TSTEP);
  fprintf(f, "# |state|  %6d\n", STATESIZE);


  fprintf(f, "n\tEobs\n");

  for (int i = 0; (i < STATESIZE); i++) {
    eigen_norm_state_alloc(evec, HSTEP, i, &(psis[i]));

    asprintf(&filename, "%s-psi%03d.txt", prefix, i);
    FILE *fpsi = fopen(filename, "w");
    free(filename);
    fwrite_vector_complex_thorough(fpsi, psis[i]);
    fclose(fpsi);
    
    fprintf(f, "%d\t%0.6f\n",
	  i, gsl_vector_get(eval, i));
  }
  
  fclose(f);

  if (0) {
    for (int i = 0; i < 10; i++) {
      printf("\033[2J\033[H");
      printf("i = %d\n", i);
      terminal_graph_abs2(psis[i], 12, 0.5);
      puts("");
      terminal_graph_phase(psis[i], 8);
      sleep(3);
    }
  }
}

#define EVOLVE_STATE 1
#define TFINAL 100.0

#define TON 20.0
#define TOFF 40.0
#define TPERSTEP (0.1 / TSTEP)

int vstep_for_tstep(int tstep) {
  const int tstep_on = tstep - (TON / TSTEP);
  const int tstep_off = tstep - (TOFF / TSTEP);
  if (tstep_on < 0) { return 0; }
  else if (tstep_off < 0) {
    const int nstep = tstep_on / TPERSTEP;
    return (nstep >= NSTEPS) ? (NSTEPS-1) : nstep;
  } else {
    const int nstep = tstep_off / TPERSTEP;
    return (NSTEPS-1) - ((nstep >= NSTEPS) ? (NSTEPS-1) : nstep);
  }
}

double v_scale_abrupt(int tstep)
{
  const double t = tstep * TSTEP; 
  if (t < TON) { return 0.0; }
  else if (t < TOFF) { return V0MAX; }
  else { return 0.0; }
}

double v_scale_stepwise(int tstep)
{
  const int tstep_on = tstep - (TON / TSTEP);
  const int tstep_off = tstep - (TOFF / TSTEP);
  if (tstep_on < 0) { return 0.0; }
  else if (tstep_off < 0) {
    const int nstep = tstep_on / WRITEEVERY;
    if (nstep >= NSTEPS) {
      return V0MAX;
    } else {
      return V0MAX * ( ((double) nstep) / ((double) NSTEPS) );
    }
  }
  else { return 0.0; }
}

void v_triangle(gsl_vector *V, double scale)
{
  int halfsize = V->size / 2;
  double istep = scale / ((double) halfsize);
  for (int i = 0; i < (V->size / 2); i++) {
    gsl_vector_set(V, i, istep * (double) i);
    gsl_vector_set(V, V->size - (i + 1), istep * (double) i);
  }
}

// Pure sine potential causes uniform phase roll
// Integral of (1 - cos th) over a full cycle = +1
// Need -0.5 over all points
void v_sin(gsl_vector *V, double scale)
{
  const int quarter = V->size / 4;
  for (int i = 0; i < quarter; i++) {
    gsl_vector_set(V, i, -0.5 * scale);
  }
  for (int i = quarter; i < 3*quarter; i++) {
    double th = M_PI * ((double) (i - quarter)) / ((double) quarter);
    gsl_vector_set(V, i, scale * (0.5 - cos(th)));
  }
  for (int i = 3*quarter; i < V->size; i++) {
    gsl_vector_set(V, i, -0.5 * scale);
  }
}

void vtstep(gsl_vector *V, int tstep)
{
  v_sin(V, v_scale_stepwise(tstep));
}

void terminal_graph_raw(const gsl_vector *v, const unsigned int nlines, char filled);

/*
    vtstep(V, tstep);
    set_hamiltonian_circular(Hprev, V, PLANCK, MASS, HSTEP);
    vtstep(V, tstep+1);
    set_hamiltonian_circular(Hnext, V, PLANCK, MASS, HSTEP);
    
    set_timeevol_halves(U, Hprev, Hnext, PLANCK, TSTEP, NULL);

    timeevol_state(psinew, U, psi);
*/

void evolve(gsl_vector *const Vs[NSTEPS], const precomputed_timeevol *Uall)
{
  gsl_vector *V = gsl_vector_calloc(STATESIZE);
  
  gsl_matrix *H0 = gsl_matrix_alloc(STATESIZE, STATESIZE);

  set_hamiltonian_circular(H0, V, PLANCK, MASS, HSTEP);

  gsl_vector *eval;
  gsl_matrix *evec;
  gsl_vector_complex *psi;
  gsl_vector_complex **psis = calloc(STATESIZE, sizeof(gsl_vector_complex *));
  
  eigen_solve_alloc(H0, &eval, &evec);
  for (int i = 0; i < STATESIZE; i++) {
    eigen_norm_state_alloc(evec, HSTEP, i, &(psis[i]));
  }

  psi = gsl_vector_complex_calloc(STATESIZE);
  
  gsl_vector_complex_memcpy(psi, psis[0]);
  
  FILE *psi1t = fopen("circdata/psi-t.txt", "w");
  
  gsl_vector_complex *psinew = gsl_vector_complex_calloc(STATESIZE);
  gsl_vector_complex *psieig = gsl_vector_complex_calloc(STATESIZE);
  
  gsl_complex hstep_c;
  GSL_SET_COMPLEX(&hstep_c, HSTEP, 0.0);

  gsl_vector *Vdisp = gsl_vector_calloc(STATESIZE);
  
  for (int tstep = 0; (tstep * TSTEP) <= TFINAL; tstep++) {
    const double t = tstep * TSTEP;

    if (tstep % WRITEEVERY == 0) {
      for (int i = 0; i < STATESIZE; i++) {
        gsl_blas_zdotc(psis[i], psi, gsl_vector_complex_ptr(psieig, i));
      }
      gsl_vector_complex_scale(psieig, hstep_c);

      gsl_vector_memcpy(Vdisp, V);
      gsl_vector_add_constant(Vdisp, V0MAX);
      gsl_vector_scale(Vdisp, 0.25 / V0MAX);
      
      printf("\033[2J\033[H");
      printf("t = %0.2f (tstep %6d)\n", t, tstep);
      terminal_graph_abs2(psi, 32, 0.5);
      puts("");
      terminal_graph_phase(psi, 8);
      puts("");
      terminal_graph_abs(psieig, 16, 1.2);
      puts("");
      terminal_graph_raw(Vdisp, 16, '%');
      
      fprintf(psi1t, "%0.6f", t);
      fwrite_vector_complex_abs2(psi1t, psi);
    }

    const int vstep_curr = vstep_for_tstep(tstep);
    const int vstep_next = vstep_for_tstep(tstep+1);

    const timeevol_halves *U;

    if (vstep_next == vstep_curr) {
      U = Uall->Usteady[vstep_curr];
    } else if (vstep_next == vstep_curr + 1) {
      U = Uall->Uupfrom[vstep_curr];
    } else if (vstep_next == vstep_curr - 1) {
      U = Uall->Udownto[vstep_next];
    } else {
      fprintf(stderr, "tstep = %d, vstep_curr = %d, vstep_next = %d\n",
	    tstep, vstep_curr, vstep_next);
      exit(1);
    }

    timeevol_state(psinew, U, psi);

    gsl_vector_complex_memcpy(psi, psinew);
  }

  fclose(psi1t);
}

precomputed_timeevol *precompute(gsl_vector *const Vs[NSTEPS])
{
  precomputed_timeevol *Uall = calloc(1, sizeof(precomputed_timeevol));

  gsl_matrix *Hprev = gsl_matrix_alloc(STATESIZE, STATESIZE);
  gsl_matrix *Hnext = gsl_matrix_alloc(STATESIZE, STATESIZE);

  for (int i = 0; i < NSTEPS; i++) {
    Uall->Usteady[i] = timeevol_halves_alloc(STATESIZE);
    
    set_hamiltonian_circular(Hprev, Vs[i], PLANCK, MASS, HSTEP);
    set_hamiltonian_circular(Hnext, Vs[i], PLANCK, MASS, HSTEP);
    set_timeevol_halves(Uall->Usteady[i], Hprev, Hnext, PLANCK, TSTEP, NULL);

    if ((i+1) < NSTEPS) {
      Uall->Uupfrom[i] = timeevol_halves_alloc(STATESIZE);
      set_hamiltonian_circular(Hnext, Vs[i+1], PLANCK, MASS, HSTEP);
      set_timeevol_halves(Uall->Uupfrom[i], Hprev, Hnext, PLANCK, TSTEP, NULL);    
      
      Uall->Udownto[i] = timeevol_halves_alloc(STATESIZE);
      set_hamiltonian_circular(Hprev, Vs[i+1], PLANCK, MASS, HSTEP);
      set_hamiltonian_circular(Hnext, Vs[i], PLANCK, MASS, HSTEP);
      set_timeevol_halves(Uall->Udownto[i], Hprev, Hnext, PLANCK, TSTEP, NULL);    
    }
  }

  gsl_matrix_free(Hprev);
  gsl_matrix_free(Hnext);

  return Uall;
}
