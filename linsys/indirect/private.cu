#include "private.h"

#include "cublas_v2.h"     // if you need CUBLAS, include before magma.h

#include <cuda_runtime.h>

#define CG_BEST_TOL 1e-9
#define CG_MIN_TOL 1e-1
#define PRINT_INTERVAL 100

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static scs_int totCgIts;
static timer linsysTimer;
static scs_float totalSolveTime;

char * getLinSysMethod(Data * d, Priv * p) {
	char * str = (char*)scs_malloc(sizeof(char) * 128);
	sprintf(str, "sparse-indirect, nnz in A = %li, CG tol ~ 1/iter^(%2.2f)", (long ) 0, d->cg_rate);
	return str;
}

char * getLinSysSummary(Priv * p, Info * info) {
	char * str = (char*)scs_malloc(sizeof(char) * 128);
	sprintf(str, "\tLin-sys: avg # CG iterations: %2.2f, avg solve time: %1.2es\n",
			(scs_float ) totCgIts / (info->iter + 1), totalSolveTime / (info->iter + 1) / 1e3);
	totCgIts = 0;
	totalSolveTime = 0;
	return str;
}

static __global__
void getPreconditioner_kernel(scs_int n, scs_float * M, scs_float * A_x, scs_int * A_p, scs_float rho) {
    scs_int i, col_len, col_start;
    scs_int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col > n) return;

    col_len = A_p[col + 1] - A_p[col];
    col_start = A_p[col];

    M[col] = 0.0;
    for (i = 0; i < col_len; ++i) {
        scs_float t = A_x[col_start + i];
        M[col] += t * t;
    }
    M[col] = 1 / (rho + M[col]);
}

/* M = inv ( diag ( RHO_X * I + A'A ) ) */
void getPreconditioner(Data *d, Priv *p) {
    scs_float * d_M;
	scs_float * M = p->M;
	AMatrix * A = d->A;

#ifdef EXTRAVERBOSE
	scs_printf("getting pre-conditioner\n");
#endif

    gpuErrchk(cudaMalloc((void**)&d_M, d->n * sizeof(scs_float)));

    dim3 threads(256);
    dim3 grid((d->n/256) + 1);

    getPreconditioner_kernel<<< grid, threads >>>
        (d->n, d_M, A->d_x, A->d_p, d->rho_x);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(M, d_M, d->n * sizeof(scs_float),
                         cudaMemcpyDeviceToHost));

#ifdef EXTRAVERBOSE
	scs_printf("finished getting pre-conditioner\n");
#endif

}

static void transpose(Data * d, Priv * p) {
	scs_int * d_Ci = p->d_Ati;
	scs_int * d_Cp = p->d_Atp;
	scs_float * d_Cx = p->d_Atx;
	scs_int m = d->n;
	scs_int n = d->m;

	scs_int * d_Ap = d->A->d_p;
	scs_int * d_Ai = d->A->d_i;
	scs_float * d_Ax = d->A->d_x;

#ifdef EXTRAVERBOSE
	timer transposeTimer;
	scs_printf("transposing A\n");
	tic(&transposeTimer);
#endif

    CUSPARSE(csr2csc)(p->cusparse, m, n, p->nnz,
                      d_Ax, d_Ap, d_Ai,
                      d_Cx, d_Ci, d_Cp,
                      CUSPARSE_ACTION_NUMERIC,
                      CUSPARSE_INDEX_BASE_ZERO);

#ifdef EXTRAVERBOSE
	scs_printf("finished transposing A, time: %1.2es\n", tocq(&transposeTimer) / 1e3);
#endif

}

void freePriv(Priv * p) {
	if (p) {
		if (p->p)
			scs_free(p->p);
		if (p->r)
			scs_free(p->r);
		if (p->Gp)
			scs_free(p->Gp);
		if (p->tmp)
			scs_free(p->tmp);
		if (p->d_Ati)
			gpuErrchk(cudaFree(p->d_Ati));
		if (p->d_Atx)
			gpuErrchk(cudaFree(p->d_Atx));
		if (p->d_Atp)
			gpuErrchk(cudaFree(p->d_Atp));
		if (p->z)
			scs_free(p->z);
		if (p->M)
			scs_free(p->M);
        if (p->descr)
            cusparseDestroyMatDescr(p->descr);
        if (p->cusparse)
             cusparseDestroy(p->cusparse);
		scs_free(p);
	}

    //magma_finalize();
}

/* solves (I+A'A)x = b, s warm start, solution stored in b */
/*y = (RHO_X * I + A'A)x */
static void matVec(Data * d, Priv * p, const scs_float * x, scs_float * y) {
	scs_float * tmp = p->tmp;
	memset(tmp, 0, d->m * sizeof(scs_float));
	accumByA(d, p, x, tmp);
	memset(y, 0, d->n * sizeof(scs_float));
	accumByAtrans(d, p, tmp, y);
	addScaledArray(y, x, d->n, d->rho_x);
}

void _accumByAtrans(Priv * p, scs_int m, scs_int n, scs_float * d_Ax, scs_int * d_Ai, scs_int * d_Ap, scs_float *d_x, scs_float *d_y, const scs_float *x, scs_float *y) {
	/* y  = A'*x
	 A in column compressed format
	 parallelizes over columns (rows of A')
	 */
    scs_float kIdent = 1.0;

    gpuErrchk(cudaMemcpy(d_x, x, n * sizeof(scs_float),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, y, m * sizeof(scs_float),
                         cudaMemcpyHostToDevice));

    cusparseStatus_t status;

    status = CUSPARSE(csrmv)(p->cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, n, p->nnz,
        &kIdent,
        p->descr,
        d_Ax, d_Ap, d_Ai,
        d_x,
        &kIdent,
        d_y);

    switch (status) {
case CUSPARSE_STATUS_SUCCESS:
    break;
case CUSPARSE_STATUS_NOT_INITIALIZED:
    printf("csrmv not init\n");
    break;
case CUSPARSE_STATUS_ALLOC_FAILED:
    printf("csrmv alloc failed\n");
    break;
case CUSPARSE_STATUS_INVALID_VALUE:
    printf("csrmv invalid value\n");
    break;
case CUSPARSE_STATUS_ARCH_MISMATCH:
    printf("csrmv arch mismatch\n");
    break;
case CUSPARSE_STATUS_MAPPING_ERROR:
    printf("csrmv mapping error\n");
    break;
case CUSPARSE_STATUS_EXECUTION_FAILED:
    printf("csrmv execution failed\n");
    break;
case CUSPARSE_STATUS_INTERNAL_ERROR:
    printf("csrmv internal error\n");
    break;
case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    printf("csrmv matrix type not supported\n");
    break;
default:
    printf("none? %d\n", status);
    break;
}

    gpuErrchk(cudaMemcpy(y, d_y, m * sizeof(scs_float),
                         cudaMemcpyDeviceToHost));
}

void accumByAtrans(Data * d, Priv * p, const scs_float *x, scs_float *y) {
	AMatrix * A = d->A;
    scs_int m, n;
    scs_float *d_x, *d_y;
    m = d->n;
    n = d->m;
    d_x = p->d_y;
    d_y = p->d_x;
	_accumByAtrans(p, m, n, A->d_x, A->d_i, A->d_p, d_x, d_y, x, y);
}

void accumByA(Data * d, Priv * p, const scs_float *x, scs_float *y) {
    scs_int m, n;
    scs_float *d_x, *d_y;
    m = d->m;
    n = d->n;
    d_x = p->d_x;
    d_y = p->d_y;
	_accumByAtrans(p, m, n, p->d_Atx, p->d_Ati, p->d_Atp, d_x, d_y, x, y);
}

static void applyPreConditioner(scs_float * M, scs_float * z, scs_float * r, scs_int n, scs_float *ipzr) {
	scs_int i;
	*ipzr = 0;
	for (i = 0; i < n; ++i) {
		z[i] = r[i] * M[i];
		*ipzr += z[i] * r[i];
	}
}

Priv * initPriv(Data * d) {
    //magma_init();
    cudaError_t cudaStat;

	AMatrix * A = d->A;
	Priv * p = (Priv*)scs_calloc(1, sizeof(Priv));
	p->p = (scs_float*)scs_malloc((d->n) * sizeof(scs_float));
	p->r = (scs_float*)scs_malloc((d->n) * sizeof(scs_float));
	p->Gp = (scs_float*)scs_malloc((d->n) * sizeof(scs_float));
	p->tmp = (scs_float*)scs_malloc((d->m) * sizeof(scs_float));

	/* preconditioner memory */
	p->z = (scs_float*)scs_malloc((d->n) * sizeof(scs_float));
	p->M = (scs_float*)scs_malloc((d->n) * sizeof(scs_float));


    cusparseCreate(&p->cusparse);
    cusparseCreateMatDescr(&p->descr);
    cusparseSetMatType(p->descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(p->descr, CUSPARSE_INDEX_BASE_ZERO);

    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(&(p->nnz), &(A->d_p[d->n]), 1 * sizeof(scs_int),
                         cudaMemcpyDeviceToHost));
    if (cudaStat != cudaSuccess) {
        printf("copy to nnz failed %d\n", cudaStat);
        printf("p->nnz: %d, A->d_p[d->n]: %d\n",
               &p->nnz, &(A->d_p[d->n]));
        exit(-1);
    }

    gpuErrchk(cudaMalloc((void**)&(p->d_Ati), (p->nnz) * sizeof(scs_int)));
    gpuErrchk(cudaMalloc((void**)&(p->d_Atp), (d->m + 1) * sizeof(scs_int)));
    gpuErrchk(cudaMalloc((void**)&(p->d_Atx), (p->nnz) * sizeof(scs_float)));
    gpuErrchk(cudaMalloc((void**)&(p->d_x), (d->n) * sizeof(scs_float)));
    gpuErrchk(cudaMalloc((void**)&(p->d_y), (d->m) * sizeof(scs_float)));

	transpose(d, p);
	getPreconditioner(d, p);
	totalSolveTime = 0;
	totCgIts = 0;

	if (!p->p || !p->r || !p->Gp || !p->tmp || !p->d_Ati || !p->d_Atp || !p->d_Atx) {
		freePriv(p);
		return NULL;
	}
	return p;
}

static scs_int pcg(Data *d, Priv * pr, const scs_float * s, scs_float * b, scs_int max_its, scs_float tol) {
	scs_int i, n = d->n;
	scs_float ipzr, ipzrOld, alpha;
	scs_float *p = pr->p; /* cg direction */
	scs_float *Gp = pr->Gp; /* updated CG direction */
	scs_float *r = pr->r; /* cg residual */
	scs_float *z = pr->z; /* for preconditioning */
	scs_float *M = pr->M; /* inverse diagonal preconditioner */

	if (s == NULL) {
		memcpy(r, b, n * sizeof(scs_float));
		memset(b, 0, n * sizeof(scs_float));
	} else {
		matVec(d, pr, s, r);
		addScaledArray(r, b, n, -1);
		scaleArray(r, -1, n);
		memcpy(b, s, n * sizeof(scs_float));
	}
	applyPreConditioner(M, z, r, n, &ipzr);
	memcpy(p, z, n * sizeof(scs_float));

	for (i = 0; i < max_its; ++i) {
		matVec(d, pr, p, Gp);

		alpha = ipzr / innerProd(p, Gp, n);
		addScaledArray(b, p, n, alpha);
		addScaledArray(r, Gp, n, -alpha);

		if (calcNorm(r, n) < tol) {
            #ifdef EXTRAVERBOSE
            scs_printf("tol: %.4e, resid: %.4e, iters: %li\n", tol, calcNorm(r, n), (long) i+1);
            #endif
			return i + 1;
		}
		ipzrOld = ipzr;
		applyPreConditioner(M, z, r, n, &ipzr);

		scaleArray(p, ipzr / ipzrOld, n);
		addScaledArray(p, z, n, 1);
	}
	return i;
}

scs_int solveLinSys(Data *d, Priv * p, scs_float * b, const scs_float * s, scs_int iter) {
	scs_int cgIts;
	scs_float cgTol = calcNorm(b, d->n) * (iter < 0 ? CG_BEST_TOL : CG_MIN_TOL / POWF((scs_float) iter + 1, d->cg_rate));

	tic(&linsysTimer);
	/* solves Mx = b, for x but stores result in b */
	/* s contains warm-start (if available) */
	accumByAtrans(d, p, &(b[d->n]), b);
	/* solves (I+A'A)x = b, s warm start, solution stored in b */
	cgIts = pcg(d, p, s, b, d->n, MAX(cgTol, CG_BEST_TOL));
	scaleArray(&(b[d->n]), -1, d->m);
	accumByA(d, p, b, &(b[d->n]));

	if (iter >= 0) {
		totCgIts += cgIts;
	}

	totalSolveTime += tocq(&linsysTimer);
#ifdef EXTRAVERBOSE
	scs_printf("linsys solve time: %1.2es\n", tocq(&linsysTimer) / 1e3);
#endif
	return 0;
}

