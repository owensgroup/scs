UNAME = $(shell uname -s)
CC = gcc
NVCC = nvcc

ifneq (, $(findstring CYGWIN, $(UNAME)))
ISWINDOWS := 1
else ifneq (, $(findstring MINGW, $(UNAME)))
ISWINDOWS := 1
else ifneq (, $(findstring MSYS, $(UNAME)))
ISWINDOWS := 1
else
ISWINDOWS := 0
endif

ifeq ($(UNAME), Darwin)
# we're on apple, no need to link rt library
LDFLAGS = -lm
SHARED = dylib
else ifeq ($(ISWINDOWS), 1)
# we're on windows (cygwin or msys)
LDFLAGS = -lm
SHARED = dll
else
# we're on a linux system, use accurate timer provided by clock_gettime()
LDFLAGS = -lm -lrt
SHARED = so
endif

CFLAGS = -g -Wall -pedantic -O3 -funroll-loops -Wstrict-prototypes -I. -Iinclude #-Wextra
ifneq ($(ISWINDOWS), 1)
CFLAGS += -fPIC
endif
NVCCFLAGS = -O3 -Xcompiler -fno-strict-aliasing -I. -Iinclude

LINSYS = linsys
DIRSRC = $(LINSYS)/direct
DIRSRCEXT = $(DIRSRC)/external
INDIRSRC = $(LINSYS)/indirect

OUT = out
AR = ar
ARFLAGS = rv
ARCHIVE = $(AR) $(ARFLAGS)
RANLIB = ranlib

CFLAGS += -I/usr/local/cuda/include -L/usr/local/cuda/lib
LDFLAGS += -lcudart -lcusparse

########### OPTIONAL FLAGS ##########
# CFLAGS += -DDLONG # use longs rather than ints
# CFLAGS += -DFLOAT # use floats rather than doubles
# CFLAGS += -DNOVALIDATE # remove data validation step
CFLAGS += -DEXTRAVERBOSE # extra verbosity level
# CFLAGS += -DNOBLASUNDERSCORE # if your blas install does not use underscores in function names 

############ OPENMP: ############
# set USE_OPENMP = 1 to allow openmp (multi-threaded matrix multiplies):
# set the number of threads to, for example, 4 by entering the command:
# export OMP_NUM_THREADS=4

USE_OPENMP = 0

ifneq ($(USE_OPENMP), 0)
  CFLAGS += -fopenmp -DOPENMP
  LDFLAGS += -lgomp
endif

############ SDPS: BLAS + LAPACK ############
# set USE_LAPACK = 1 below to enable solving SDPs
# NB: point the libraries to the locations where
# you have blas and lapack installed

USE_LAPACK = 0

ifneq ($(USE_LAPACK), 0)
  # edit these for your setup:
  LDFLAGS += -lblas -llapack #-lgfortran
  CFLAGS += -DLAPACK_LIB_FOUND
  # CFLAGS += -DBLAS64 # if blas/lapack lib uses 64 bit ints
endif
