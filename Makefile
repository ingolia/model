CC=gcc
CFLAGS=-Wall -O3 -g -I/usr/local/include/ -Ishared/
LDFLAGS=-g -L/usr/local/lib/
LOADLIBES=-lgsl -lblas -lm

SHARED_HEADERS=shared/params.h shared/schdebug.h shared/schdisplay.h shared/schutil.h shared/schutil_1d.h shared/timeevol.h
SHARED_OBJS=shared/timeevol.o shared/schdebug.o shared/schdisplay.o shared/schutil.o shared/schutil_1d.o

RING_SIX_HEADERS=ring_six/potential.h ring_six/ring_six.h
RING_SIX_OBJS=ring_six/potential.o ring_six/ring_six.o

PROGS=ring_six/ring_six

all: $(SHARED_OBJS) $(PROGS)

$(PROGS): %: $(SHARED_OBJS)

$(addsuffix .o, $(PROGS)): %.o: $(SHARED_HEADERS)

$(SHARED_OBJS): %.o: $(SHARED_HEADERS)

ring_six/ring_six: $(RING_SIX_OBJS)

$(RING_SIX_OBJS): %.o: $(RING_SIX_HEADERS)
