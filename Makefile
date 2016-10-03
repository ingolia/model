CC=gcc
CFLAGS=-O3 -g
LDFLAGS=-g
LOADLIBES=-lgsl -lblas -lm

all: d1p1 tvar tetest

d1p1: d1p1.o schutil.o

tvar: tvar.o schutil.o

tetest: tetest.o schutil.o

d1p1.o: d1p1.c schutil.h

tvar.o: tvar.c schutil.h

tetest.o: tetest.c schutil.h

schutil.o: schutil.c schutil.h
