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

#define WRITEEVERY 256

#define NSTEPS 32

typedef struct {
  timeevol_halves *Usteady[NSTEPS];
  timeevol_halves *Uupfrom[NSTEPS];
  timeevol_halves *Udownto[NSTEPS];
} fade_timeevol;

void test_6pts(void);
void solve_stationary(const gsl_vector *, gsl_matrix *);
void evolve(gsl_vector *const Vs[NSTEPS], gsl_matrix *const Hs[NSTEPS], const fade_timeevol *Uall);

fade_timeevol *make_fade_timeevol(gsl_matrix *const Hs[NSTEPS]);

void v_norm_ground(gsl_vector *V);
void v_sin_6pt(gsl_vector *V, double Vpts[6]);

int main(void)
{
  //  test_6pts();

  double Vpts[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  
  gsl_vector **V1s = calloc(NSTEPS, sizeof(gsl_vector *));
  gsl_matrix **H1s = calloc(NSTEPS, sizeof(gsl_matrix *));
  for (int i = 0; i < NSTEPS; i++) {
    V1s[i] = gsl_vector_calloc(STATESIZE);
    Vpts[0] = Vpts[2] = Vpts[2] = Vpts[3] = Vpts[4] = Vpts[5] = 0.0;
    Vpts[1] = V0MAX * ((double) i) / ((double) (NSTEPS - 1));
    v_sin_6pt(V1s[i], Vpts);
    v_norm_ground(V1s[i]);
    H1s[i] = gsl_matrix_alloc(STATESIZE, STATESIZE);
    set_hamiltonian_circular(H1s[i], V1s[i], PLANCK, MASS, HSTEP);
  }

  gsl_vector **V2s = calloc(NSTEPS, sizeof(gsl_vector *));
  gsl_matrix **H2s = calloc(NSTEPS, sizeof(gsl_matrix *));
  for (int i = 0; i < NSTEPS; i++) {
    V2s[i] = gsl_vector_calloc(STATESIZE);
    Vpts[0] = Vpts[1] = Vpts[2] = Vpts[3] = Vpts[4] = Vpts[5] = 0.0;
    Vpts[2] = V0MAX * ((double) i) / ((double) (NSTEPS - 1));
    v_sin_6pt(V2s[i], Vpts);
    v_norm_ground(V2s[i]);
    H2s[i] = gsl_matrix_alloc(STATESIZE, STATESIZE);
    set_hamiltonian_circular(H2s[i], V2s[i], PLANCK, MASS, HSTEP);
  }

  solve_stationary(V1s[0], H1s[0]);
  solve_stationary(V1s[NSTEPS-1], H1s[NSTEPS-1]);
  
  fade_timeevol *Ufade1 = make_fade_timeevol(H1s);
  fade_timeevol *ufade2 = make_fade_timeevol(H2s);

  evolve(V1s, H1s, Ufade1);
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
			         
  for (int i = 0; i < 4; i++) {
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

#define TPERSTEP ((1.0 / 32.0) / TSTEP)

int vstep_for_tstep(double Ton, double Toff, int tstep) {
  const int tstep_on = tstep - (Ton / TSTEP);
  const int tstep_off = tstep - (Toff / TSTEP);
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

#define TFADE1 (10.0)

void evolve(gsl_vector *const Vs[NSTEPS], gsl_matrix *const Hs[NSTEPS], const fade_timeevol *Ufade1)
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

  FILE *f = fopen("circdata/evolve-states.txt", "w");
  
  if (0) {
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
    const int vstep_curr = vstep_for_tstep(5.0, 25.0, tstep);

    if (tstep % WRITEEVERY == 0) {
      fprintf(f, "%d", tstep);
      for (int i = 0; i < STATESIZE; i++) {
        gsl_blas_zdotc(psis[i], psi, gsl_vector_complex_ptr(psieig, i));
        fprintf(f, "\t%0.6f", gsl_complex_abs(gsl_vector_complex_get(psieig, i))*HSTEP);
      }
      fprintf(f, "\n");
      fflush(f);
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

    const int vstep_next = vstep_for_tstep(5.0, 25.0, tstep+1);

    const timeevol_halves *U;

    if (vstep_next == vstep_curr) {
      U = Ufade1->Usteady[vstep_curr];
    } else if (vstep_next == vstep_curr + 1) {
      U = Ufade1->Uupfrom[vstep_curr];
    } else if (vstep_next == vstep_curr - 1) {
      U = Ufade1->Udownto[vstep_next];
    } else {
      fprintf(stderr, "tstep = %d, vstep_curr = %d, vstep_next = %d\n",
	    tstep, vstep_curr, vstep_next);
      exit(1);
    }

    timeevol_state(psinew, U, psi);

    gsl_vector_complex_memcpy(psi, psinew);
  }
}

fade_timeevol *make_fade_timeevol(gsl_matrix *const Hs[NSTEPS])
{
  fade_timeevol *Uall = calloc(1, sizeof(fade_timeevol));

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

// Verifying constants, etc.
// \psi_1(x) = a sin(x)
// <\psi_1'|\psi_1> = a^2 \int_0^{2\pi} sin^2(x)
//   = a^2 \int_0^{2\pi} 1/2 (1 - cos(2x))
//   = a^2 1/2 (2\pi)
//   = a^2 \pi
// and so a = 1/sqrt(pi) = 1.0 / M_SQRTPI 
// <\psi_1'|H|\psi_1> = \int_0^{2\pi} dx a sin (x) (-\hbar^2)/(2m) (a sin(x))''
//   =  \int_0^{2\pi} dx a sin(x) (-\hbar^2)/(2m) (-a sin(x))
//   =  \int_0^{2\pi} dx a^2 (\hbar^2)/(2m) sin^2(x)
//   =  \int_0^{2\pi} dx (1/\pi) (\hbar^2)/(2m) (1/2) (1 - cos(2x))
//   = (1/\pi) (\hbar^2)/(2m) (1/2) (2\pi) - \int_0^{2\pi} dx ... cos(2x)
//   = (\hbar^2)/(2m)
// And for a sin(kx), a=(\pi)^{-1/2} and E = k^2 E_1
// Frequency, E \Psi(t) = i \hbar \Psi(t)'
//   \Psi(t)' = (- i E / \hbar) \Psi(t)
//   \Psi(t) = \Psi(0) exp(-i (E / \hbar) t)
//   \omega = E / \hbar and \omega_1 = \hbar/(2m)
// Bigger \hbar or smaller m => faster frequency
// \hbar 4.0 and m 1.0 means \omega 2 and period 4 \pi
