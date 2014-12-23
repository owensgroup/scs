#ifndef AMATRIX_H_GUARD
#define AMATRIX_H_GUARD

/* this struct defines the data matrix A */
struct A_DATA_MATRIX {
	/* A is supplied in column compressed format */
	scs_float * d_x;  /* A values, size: NNZ A */
	scs_int * d_i;    /* A row index, size: NNZ A */
	scs_int * d_p;    /* A column pointer, size: n+1 */
};

#endif
