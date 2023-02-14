#ifndef CUDA_LIKE_SEQ
#define CUDA_LIKE_SEQ

/**
 * This is a special kind of batch matrix multiplication, in
 * which the same two matrices are multiplied, but the values
 * of the two matrices are filtered under a mask that differs
 * across the batch.
 * Input arrays with dimensions:
 *    A : [K][N]float
 *    B : [N][K]float
 *    Xt: [N][M]char  (the transposed mask)
 * Result:
 *    Y : [M][K][K]float
 * T is some numeric type (single/double precision floats).
 * 
 * Temporal locality:
 *    the index of each array read is invariant to two parallel dimensions.
 *
 * Changes w.r.t. goldenSeq.h:
 *   1. We pass Xt, the transposed of X, as argument otherwise
 *       we would have poor spatial locality (on GPU) when reading X.
 *   2. The loop of index "i" of count "M" is split (stripmined into)
 *       two loops: one of index "ii", which goes with a stride "T"
 *       and is kept outermost, and another one of index "i", which
 *       is essentially moved in the innermost position in the nest.
 **/

//#define CUDA_DEMO

template<int T>
void cudaLikeSeq( float* A, float* B, char* X, float* Y
                , const int M, const int K, const int N
) {
#ifdef CUDA_DEMO
    unsigned long long mem_size_X = sizeof(char) * M * N;
    char*  Xt = (char*) malloc(mem_size_X);
    runSeqTranspose<char,256>(X, Xt, M, N);
#endif

    #pragma omp parallel for schedule(static) 
    for(int ii=0; ii<M; ii+=T) { // parallel (on Grid)
        for(int j1=0; j1<K; j1++) { // parallel (on Block.y)
            for(int j2=0; j2<K; j2++) { // parallel (on Block.x)
                // implying the remapping of (a chunk of) C to scratchpad memory
#ifdef CUDA_DEMO
                char Xsh[T];
#endif

                // each "thread" computes T elements kept in "register" memory,
                // i.e., array "acc" can be scalarized.
                float acc[T];
                for(int i=0; i<T; i++)
                    acc[i] = 0.0;

                for(int q=0; q<N; q++) { // sequential (reduction)
                    float a = A[j1*N + q];
                    float b = B[q*K + j2];

                    // Remaps/copies the slice X[ii:min(ii+T,M), q] that
                    //   is used below from global to scratchpad memory. 
                    // In Cuda, this should be a collective copy,
                    //   i.e., the first T threads of a Cuda block,
                    //   each copy one element.
                    // In this sequential code, we will use a loop.
#ifdef CUDA_DEMO
                    for(int i=0; i<T; i++) {
                        char v = 0;
                        if(ii+i < M) {
                            v = Xt[q*M + (ii+i)];
                        }
                        Xsh[i] = v;
                    }
#endif
                    // finally the computation of the result:
                    for(int i=0; i<T && i+ii<M; i++) {
                        float v = 0.0;
#ifdef CUDA_DEMO
                        if( Xsh[i] != 0 )
                            v = 1.0;
#else
                        if( X[(i+ii)*N + q] != 0 )
                            v = 1.0;
#endif
                        acc[i] += a*b*v;
                    }
                }

                // update the result in global memory
                for(int i=0; i<T; i++) {
                    if( (ii+i) < M )
                        Y[(ii+i)*K*K + j1*K + j2] = acc[i];
                }
            }
        }
    }
#ifdef CUDA_DEMO
    free(Xt);
#endif
}

#endif
