#ifndef BMMM_KERNELS
#define BMMM_KERNELS

/**
 * We use ElTp for the (generic) array-element type.
 * For simplicity, this kernel assumes a grid of M blocks
 * (on the x dimension) and that the block size on the x and
 * y dimensions is K, i.e., blockDim.x == blockDim.y == K. 
 */

template <class ElTp> __global__ 
void bmmmNaiveKer ( ElTp* A, ElTp* B, char* X, ElTp* Y
                  , const int M,  const int K, const int N
) {
  const int i  = blockIdx.x;
  const int j1 = threadIdx.y;
  const int j2 = threadIdx.x;

  // sanity checks
  if( (i >= M) || (j1 >= K) )
    return;

  ElTp acc = 0.0f;

  for(int q=0; q<N; q++) { // reduction
    float v = 0.0;
    if( X[i*N + q] != 0 ) {
      v = 1.0;
    }
    ElTp a = A[j1*N + q];
    ElTp b = B[q*K + j2];
    acc += a*b*v;
  }
  Y[i*K*K + j1*K + j2] = acc;
}


/**
 * This is not the optimal version as it
 *   does not efficiently sequentializes.
 * Assumes:
 *    blockDim.y == blockDim.x == T
 */
template <class ElTp, int T> 
__global__ void matTransposeTiledKer(ElTp* A, ElTp* A_tr, const int heightA, const int widthA) {
  __shared__ ElTp tile[T][T+1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if( x < widthA && y < heightA )
      tile[threadIdx.y][threadIdx.x] = A[y*widthA + x];

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x; 
  y = blockIdx.x * T + threadIdx.y;

  if( x < heightA && y < widthA )
      A_tr[y*heightA + x] = tile[threadIdx.x][threadIdx.y];
}


/**
 * ElTp is some numeric type, e.g., float.
 * T is the size of the tile used for the
 *    outermost loop of count M.
 * 
 * Array dimensions are as in goldenSeq;
 * X_tr is the transposed on X.
 * A:[K][N] , B:[N][K] , X_tr:[N][M] , Y:[M][K][K]
 * Assumes:
 *    (1) blockDim.y * blockDim.x >= R
 *    (2) blockDim.x == blockDim.y == K
 */
template <class ElTp, int T> __global__
void bmmmTiledKer ( ElTp* A,      ElTp* B, char* X_tr,   ElTp* Y
                  , const int M,  const int K, const int N
) {
  __shared__ ElTp Xsh_tr[R];
  ElTp acc[R];

  const int ii  = blockIdx.x;
  const int j1  = threadIdx.y;
  const int j2  = threadIdx.x;
  const int i   = ii * R;
  const int flat_thid = threadIdx.y * K + threadIdx.x;

  #pragma unroll
  for(int s=0; s<R; s++)
    acc[s] = 0;

  /***********************************************
   *** Cuda Exercise 1: ***
   * 
   * With the help of the pseudocode from the
   * lecture slides, please implement the rest of
   * the code of this kernel.
   * Remember to flatten the indices to all arrays
   * hold in global memory, i.e., A, B, X_tr, Y.
   ***********************************************/
  
}
#endif
