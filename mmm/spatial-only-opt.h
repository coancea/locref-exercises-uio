#ifndef SPATIAL_OPT
#define SPATIAL_OPT

/**
 * Computes matrix multiplication C = A*B
 * Semantically the matrix sizes are:
 *    A : [heightA][widthA]ElTp
 *    B : [ widthA][widthB]ElTp
 *    C : [heightA][widthB]ElTp
 *  for some numeric type ElTp.
 **/
template<class ElTp>
void spatialOpt(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
    ElTp* B_tr = (ElTp*)malloc( widthA * widthB * sizeof(ElTp) );

    runSeqTranspose<ElTp,16>(B, B_tr, widthA, widthB);

    #pragma omp parallel for schedule(static) collapse(2)
    for(int i=0; i<heightA; i++) {
        for(int j=0; j<widthB; j++) {
            ElTp c = 0;
            for(int k=0; k<widthA; k++) {
                c += A[i*widthA +k] * B_tr[j*widthA + k];
            }
            C[i*widthB + j] = c;
        }
    }

    free(B_tr);
}

#endif
