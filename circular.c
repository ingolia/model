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

#include "schutil.h"
#include "writing.h"

#define NPTS 192
#define STATESIZE NPTS

#define PLANCK 4.0
#define MASS   1.0
#define V0MAX  8.0
#define LENGTH (2.0 * M_PI)
#define HSTEP (LENGTH / ((double) NPTS))
#define TSTEP (1.0/4096.0)

// psi1(x) = a * sin(x)
// <psi1'|psi1> = a^2 * M_PI => a = 1/sqrt(pi) = 1.0 / M_SQRTPI 

#define WRITEEVERY 512

#define NSTEPS 32

typedef struct {
  timeevol_halves *Usteady[NSTEPS];
  timeevol_halves *Uupfrom[NSTEPS];
  timeevol_halves *Udownto[NSTEPS];
} precomputed_timeevol;

void test_6pts(void);
void solve_stationary(const gsl_vector *, gsl_matrix *);
void evolve(gsl_vector *const Vs[NSTEPS], gsl_matrix *const Hs[NSTEPS], const precomputed_timeevol *Uall);

precomputed_timeevol *precompute(gsl_matrix *const Hs[NSTEPS]);

void v_norm_ground(gsl_vector *V);
void v_sin_6pt(gsl_vector *V, double Vpts[6]);
void v_sin(gsl_vector *V, double scale);

int main(void)
{
  //  test_6pts();
  
  gsl_vector **Vs = calloc(NSTEPS, sizeof(gsl_vector *));
  gsl_matrix **Hs = calloc(NSTEPS, sizeof(gsl_matrix *));
  for (int i = 0; i < NSTEPS; i++) {
    Vs[i] = gsl_vector_calloc(STATESIZE);
    v_sin(Vs[i], V0MAX * ((double) i) / ((double) (NSTEPS - 1)));
    v_norm_ground(Vs[i]);
    Hs[i] = gsl_matrix_alloc(STATESIZE, STATESIZE);
    set_hamiltonian_circular(Hs[i], Vs[i], PLANCK, MASS, HSTEP);
  }

  solve_stationary(Vs[0], Hs[0]);
  solve_stationary(Vs[NSTEPS-1], Hs[NSTEPS-1]);
  
  precomputed_timeevol *Uall = precompute(Hs);

  evolve(Vs, Hs, Uall);
}

void test_6pts(void)
{
  gsl_vector **Vs = calloc(64, sizeof(gsl_vector *));
  gsl_matrix **Hs = calloc(64, sizeof(gsl_vector *));

  double Vpts[6];
  
  for (int i = 0; i < 64; i++) {
    Vpts[0] = (i & 0x01) ? V0MAX : 0.0;
    Vpts[1] = (i & 0x02) ? V0MAX : 0.0;
    Vpts[2] = (i & 0x04) ? V0MAX : 0.0;
    Vpts[3] = (i & 0x08) ? V0MAX : 0.0;
    Vpts[4] = (i & 0x10) ? V0MAX : 0.0;
    Vpts[5] = (i & 0x20) ? V0MAX : 0.0;

    Vs[i] = gsl_vector_calloc(STATESIZE);
    v_sin_6pt(Vs[i], Vpts);
    v_norm_ground(Vs[i]);

    Hs[i] = gsl_matrix_calloc(STATESIZE, STATESIZE);
    solve_stationary(Vs[i], Hs[i]);
  }
}

void solve_stationary(const gsl_vector *V, gsl_matrix *H)
{
  set_hamiltonian_circular(H, V, PLANCK, MASS, HSTEP);
  
  gsl_vector *eval;
  gsl_matrix *evec;

  gsl_vector_complex **psis;
  psis = malloc(sizeof(gsl_vector_complex *) * STATESIZE);
  
  eigen_solve_alloc(H, &eval, &evec);

  double *ts = calloc(STATESIZE, sizeof(double));
  double **targs = calloc(STATESIZE, sizeof(double *));
  for (int i = 0; i < STATESIZE; i++) {
    targs[i] = calloc(STATESIZE, sizeof(double));
  }
  
  for (int i = 0; (i < STATESIZE); i++) {
    eigen_norm_state_alloc(evec, HSTEP, i, &(psis[i]));
  }
  
  gsl_vector *Vdisp = gsl_vector_alloc(V->size);
			         
  for (int i = 0; i < 8; i++) {
    double E = get_energy(H, HSTEP, psis[i]);

    printf("\033[2J\033[H");
    printf("i = %d\nE = %0.2f\n", i, E);
    terminal_graph_abs2(psis[i], 12, 0.5);
    puts("");
    terminal_graph_phase(psis[i], 8);

    puts("");
    gsl_vector_memcpy(Vdisp, V);
    gsl_vector_add_constant(Vdisp, V0MAX);
    gsl_vector_scale(Vdisp, 0.25 / V0MAX);
    terminal_graph_raw(Vdisp, 16, '%');

    sleep(1);
  }

  gsl_vector_free(Vdisp);
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

void v_sin_6pt(gsl_vector *V, double Vpts[6])
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

void v_norm_ground(gsl_vector *V)
{
  gsl_vector *eval = gsl_vector_alloc(V->size);
  
  gsl_matrix *H = gsl_matrix_alloc(V->size, V->size);
  set_hamiltonian_circular(H, V, PLANCK, MASS, HSTEP);
  
  gsl_eigen_symm_workspace *w = gsl_eigen_symm_alloc(V->size);
  gsl_eigen_symm(H, eval, w);
  gsl_eigen_symm_free(w);
  gsl_matrix_free(H);

  double Eg = gsl_vector_min(eval);
  gsl_vector_free(eval);

  gsl_vector_add_constant(V, -Eg);
}

// Integral of (1 - cos th) over a full cycle = +1
// Need -(1.0/3.0) over all points
void v_sin(gsl_vector *V, double scale)
{
  const int sixth = V->size / 6;
  for (int i = 0; i < sixth; i++) {
    gsl_vector_set(V, i, -(1.0/3.0) * scale);
  }
  for (int i = sixth; i < 3*sixth; i++) {
    double th = M_PI * ((double) (i -sixth)) / ((double) sixth);
    gsl_vector_set(V, i, scale * ((2.0/3.0) - cos(th)));
  }
  for (int i = 3*sixth; i < V->size; i++) {
    gsl_vector_set(V, i, -(1.0/3.0) * scale);
  }
}

/*
    vtstep(V, tstep);
    set_hamiltonian_circular(Hprev, V, PLANCK, MASS, HSTEP);
    vtstep(V, tstep+1);
    set_hamiltonian_circular(Hnext, V, PLANCK, MASS, HSTEP);
    
    set_timeevol_halves(U, Hprev, Hnext, PLANCK, TSTEP, NULL);

    timeevol_state(psinew, U, psi);
*/

void evolve(gsl_vector *const Vs[NSTEPS], gsl_matrix *const Hs[NSTEPS], const precomputed_timeevol *Uall)
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
  
  if (1) {
    gsl_vector_complex_memcpy(psi, psis[0]);
  } else {
    gsl_vector_complex *psi0 = gsl_vector_complex_calloc(STATESIZE);
    gsl_vector_complex_memcpy(psi0, psis[0]);
    gsl_vector_complex_scale(psi0, gsl_complex_rect(M_SQRT1_2, 0.0));

    gsl_vector_complex *psi1 = gsl_vector_complex_calloc(STATESIZE);
    gsl_vector_complex_memcpy(psi1, psis[1]);
    gsl_vector_complex_scale(psi1, gsl_complex_rect(0.5, 0.0));

    gsl_vector_complex *psi2 = gsl_vector_complex_calloc(STATESIZE);
    gsl_vector_complex_memcpy(psi2, psis[2]);
    gsl_vector_complex_scale(psi2, gsl_complex_rect(0.0, 0.5));

    gsl_vector_complex_memcpy(psi, psi0);
    gsl_vector_complex_add(psi, psi1);
    gsl_vector_complex_add(psi, psi2);
  }
    
  gsl_vector_complex *psinew = gsl_vector_complex_calloc(STATESIZE);
  gsl_vector_complex *psieig = gsl_vector_complex_calloc(STATESIZE);
  
  gsl_complex hstep_c;
  GSL_SET_COMPLEX(&hstep_c, HSTEP, 0.0);

  gsl_vector *Vdisp = gsl_vector_calloc(STATESIZE);
  
  for (int tstep = 0; (tstep * TSTEP) <= TFINAL; tstep++) {
    const double t = tstep * TSTEP;
    const int vstep_curr = vstep_for_tstep(tstep);

    if (tstep % WRITEEVERY == 0) {
      for (int i = 0; i < STATESIZE; i++) {
        gsl_blas_zdotc(psis[i], psi, gsl_vector_complex_ptr(psieig, i));
      }
      gsl_vector_complex_scale(psieig, hstep_c);

      gsl_vector_memcpy(Vdisp, Vs[vstep_curr]);
      gsl_vector_add_constant(Vdisp, V0MAX);
      gsl_vector_scale(Vdisp, 0.25 / V0MAX);

      double E = get_energy(Hs[vstep_curr], HSTEP, psi);
      
      printf("\033[2J\033[H");
      printf("t = %0.2f (tstep %6d), E = %0.2f\n", t, tstep, E);
      terminal_graph_abs2(psi, 32, 0.5);
      puts("");
      terminal_graph_phase(psi, 8);
      puts("");
      terminal_graph_abs(psieig, 8, 1.2);
      puts("");
      terminal_graph_raw(Vdisp, 16, '%');
    }

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
}

precomputed_timeevol *precompute(gsl_matrix *const Hs[NSTEPS])
{
  precomputed_timeevol *Uall = calloc(1, sizeof(precomputed_timeevol));

  for (int i = 0; i < NSTEPS; i++) {
    Uall->Usteady[i] = timeevol_halves_alloc(STATESIZE);
    
    set_timeevol_halves(Uall->Usteady[i], Hs[i], Hs[i], PLANCK, TSTEP, NULL);

    if ((i+1) < NSTEPS) {
      Uall->Uupfrom[i] = timeevol_halves_alloc(STATESIZE);
      set_timeevol_halves(Uall->Uupfrom[i], Hs[i], Hs[i+1], PLANCK, TSTEP, NULL);    
      
      Uall->Udownto[i] = timeevol_halves_alloc(STATESIZE);
      set_timeevol_halves(Uall->Udownto[i], Hs[i+1], Hs[i], PLANCK, TSTEP, NULL);    
    }
  }

  return Uall;
}
