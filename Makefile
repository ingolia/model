CC=gcc
CFLAGS=-O3 -g -I/usr/local/include/
LDFLAGS=-g -L/usr/local/lib/
LOADLIBES=-lgsl -lblas -lm

HEADERS=schutil.h writing.h grid2d.h
OBJS=schutil.o writing.o grid2d.o

PROGS=d1p1 tvar tetest stationary grid2d_test coherent

all: $(PROGS)

$(PROGS): %: $(OBJS)

$(addsuffix .o, $(PROGS)): %.o: $(HEADERS)

$(OBJS): %.o: $(HEADERS)
