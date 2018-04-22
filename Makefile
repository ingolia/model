CC=gcc
CFLAGS=-O3 -g -I/usr/local/include/
LDFLAGS=-g -L/usr/local/lib/
LOADLIBES=-lgsl -lblas -lm

SHARED_HEADERS=shared/params.h shared/schdebug.h shared/timeevol.h
SHARED_OBJS=shared/timeevol.o shared/schdebug.o

PROGS=

all: $(SHARED_OBJS) $(PROGS)

$(PROGS): %: $(SHARED_OBJS)

$(addsuffix .o, $(PROGS)): %.o: $(HEADERS)

$(SHARED_OBJS): %.o: $(HEADERS)
