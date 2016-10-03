CC=gcc
CFLAGS=-O3 -g
LDFLAGS=-g
LOADLIBES=-lgsl -lblas -lm

all: d1p1 tvar tetest

d1p1: d1p1.o schutil.o writing.o

tvar: tvar.o schutil.o writing.o

tetest: tetest.o schutil.o writing.o

d1p1.o: d1p1.c schutil.h

tvar.o: tvar.c schutil.h

tetest.o: tetest.c schutil.h writing.h

schutil.o: schutil.c schutil.h writing.h
