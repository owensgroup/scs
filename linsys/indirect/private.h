#ifndef PRIV_H_GUARD
#define PRIV_H_GUARD

#include "glbopts.h"
#include "scs.h"
#include "cs.h"
#include <math.h>
#include "linsys/common.h"
#include "linAlg.h"

#include <cusparse.h>
#include <cublas_v2.h>

#ifndef FLOAT
#define CUBLAS(x) cublas ## D ## x
#define CUSPARSE(x) cusparse ## D ## x
#else
#define CUBLAS(x) cublas ## S ## x
#define CUSPARSE(X) cusparse ## S ## x
#endif

struct PRIVATE_DATA {
    scs_int nnz; /* number of nonzeros in A */
	scs_float * p; /* cg iterate  */
	scs_float * r; /* cg residual */
	scs_float * Gp;
	scs_float * tmp;
	scs_float * d_Atx;
	scs_int * d_Ati;
	scs_int * d_Atp;
    scs_float * d_x; /* length n */
    scs_float * d_y; /* length m */
	/* preconditioning */
	scs_float * z;
	scs_float * M;
    cublasHandle_t cublas;
    cusparseHandle_t cusparse;
    cusparseMatDescr_t descr;
};

#endif
