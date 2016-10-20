CC=gcc
CFLAGS=-O3 -g -I/usr/local/include/
LDFLAGS=-g -L/usr/local/lib/
LOADLIBES=-lgsl -lblas -lm

all: d1p1 tvar tetest stationary grid2d_test

d1p1: d1p1.o schutil.o writing.o

tvar: tvar.o schutil.o writing.o

tetest: tetest.o schutil.o writing.o

stationary: stationary.o schutil.o writing.o

d1p1.o: d1p1.c schutil.h

tvar.o: tvar.c schutil.h writing.h

tetest.o: tetest.c schutil.h writing.h

stationary.o: stationary.c schutil.h writing.h

schutil.o: schutil.c schutil.h writing.h

grid2d.o: grid2d.c grid2d.h

grid2d_test.o: grid2d_test.c grid2d.h

grid2d_test: grid2d_test.o grid2d.o
