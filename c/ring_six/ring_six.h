#ifndef _RING_SIX_H
#define _RING_SIX_H 1

#define MASS   1.0
#define V0MAX  8.0
#define LENGTH (2.0 * M_PI)
#define PLANCK 4.0

#define NPTS 36
#define STATESIZE NPTS
#define HSTEP (LENGTH / ((double) NPTS))
#define TSTEP (1.0/4096.0)

#endif /* defined(_RING_SIX_H) */
