#ifndef GOLDEN
#define GOLDEN

/**
 * This is a special kind of batch matrix multiplication, in
 * which the same two matrices are multiplied, but the values
 * of the two matrices are filtered under a mask that differs
 * across the batch.
 * Input arrays with dimensions:
 *    A : [K][N]float
 *    B : [N][K]float
 *    X : [M][N]char  (the mask)
 * Result:
 *    Y : [M][K][K]float
 * T is some numeric type (single/double precision floats).
 * 
 * Temporal locality:
 *    the index of each array read is invariant to two parallel dimensions.
 **/
void goldenSeq( float* A, float* B, char* X, float* Y
              , const int M, const int K, const int N
) {
    #pragma omp parallel for schedule(static) 
    for(int i=0; i<M; i++) { // parallel
        for(int j1=0; j1<K; j1++) { // parallel
            for(int j2=0; j2<K; j2++) { // parallel
                float acc = 0.0;
                for(int q=0; q<N; q++) { // sequential (reduction)
                    float a = A[j1*N + q];
                    float b = B[q*K + j2];
                    float v = 0.0;
                    if( X[i*N + q] != 0 ) {
                        v = 1.0;
                    }
                    acc += a*b*v;
                }
                Y[i*K*K + j1*K + j2] = acc;
            }
        }
    }
}

#endif
