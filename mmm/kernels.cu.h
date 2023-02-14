#ifndef MULT_KERNELS
#define MULT_KERNELS

// widthA = heightB
template <class ElTp> 
__global__ void mmmNaiveKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  if( (gidx >= widthB) || (gidy >= heightA) ) return;

  for(int k = 0; k < widthA; k++) {
      accum += A[gidy*widthA + k] * B[k*widthB + gidx];
  }

  C[gidy*widthB + gidx] = accum;
}

/************************************************/
/*** Block+Register Tile with different tiles ***/
/***  the parallel dimensions and the seq one.***/
/*** Assumes: Ty==blockDim.y & Tx==blockDim.x ***/
/*** Hence 0 <= threadIdx.y < Ty and.         ***/
/***.      0 <= threadIdx.x < Tx              ***/
/************************************************/

template <class ElTp, int Ty, int Ry, int Tx, int Rx, int Tk>
__global__ void mmmSymBlkRegInnSeqKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {

  // remapping (a slice of) A to shared memory
  __shared__ ElTp Aloc[Ty*Ry][Tk+1];

  // remapping (a slice of) B to shared memory
  __shared__ ElTp Bloc[Tk][Tx*Rx+1]; 

  // the thread result is computed in register memory
  // and the global-memory array C is updated at the end.
  ElTp css[Ry][Rx];

  unsigned int iii = blockIdx.y * Ty * Ry;
  unsigned int jjj = blockIdx.x * Tx * Rx;

  // initialize the result with zero
  // (the neutral element for addition)
  #pragma unroll
  for(int i=0; i<Ry; i++)
      #pragma unroll
      for(int j=0; j<Rx; j++)
          css[i][j] = 0.0;

  for(int kk = 0; kk < widthA; kk += Tk) {

      /***************************************
       * Cuda Exercise 3:
       * Task 3.1:
       *   Insert here the Cuda code that collectively copies---with 
       *     all the T x T threads of the Cuda block the slices---the
       *     corresponding slice from A and B  (hold in global memory)
       *     into shared-memory arrays Aloc[T*R][T] and Bloc[T][T*R].
       *   You may assume that T = Tx = Ty = Tk and R = Rx = Ry,
       *   i.e., Tx, Ty, Tk have the same value and 
       *         Rx and Ry have the same value.
       ***************************************************************/

      /***************************************
       * Subtask 3.1.1:
       * Please insert here the code that collectively copies 
       *    with the Ty x Tx threads of the Cuda block the 
       *    following slice of A: 
       *         A[iii : iii + Ty*Ry][kk : kk+Tk] 
       *    to the shared-memory buffer Aloc, such that all accesses 
       *    to A (and Aloc) are coalesced, i.e., consecutive
       *    threads on the threadIdx.x dimension having the same
       *    threadIdx.y value read consecutive locations in memory!
       * Hints:
       * 1. Remember that threadIdx.y = 0 ... Ty-1 and, for simplicity
       *      assume that Tx = Tk, hence threadIdx.x = 0 ... Tk-1
       *    It follows that you essentially need a loop having Ry
       *      iterations, i.e., each iteration copies Ty x Tk 
       *      elements (one element per thread).
       * 2. When you read from A, remember to check that the index
       *      is within legal bound, i.e., the row-index is less than
       *      heightA and the column index is less than widthA
       * 3. Remember that A is a flat one-dimensional array, hence you
       *      need to flatten the indexing, but
       *      Aloc is a two-dimensional array 
       *      (see definitiona above at the beginning of kernel).
       **************************************************************/
      
       // Please implement Task 3.1.1 here

      /***************************************
       * Subtask 3.1.2:
       * Please insert here the code that collectively copies 
       *    with the Ty x Tx threads of the Cuda block the
       *    following slice of B: 
       *         B[kk : kk+Tk][jjj : jjj + Tx*Rx] 
       *    to the shared-memory buffer Bloc, such that all accesses 
       *    to B (and Bloc) are coalesced, i.e., consecutive
       *    threads on the threadIdx.x dimension having the same
       *    threadIdx.y value read consecutive locations in memory!
       * Hints:
       * 1. Remember that threadIdx.x = 0 ... Tx-1 and, for simplicity
       *      assume that Ty = Tk, hence threadIdx.y = 0 ... Tk-1
       *    It follows that you essentially need a loop having Rx
       *      iterations, each iteration copying Tk x Tk elements
       *      (one element per thread).
       * 2. When you read from B, remember to check that the index
       *      is within legal bound, i.e., the row-index is less than
       *      widthB and the column index is less than widthA
       * 3. Remember that B is a flat one-dimensional array, hence you
       *      need to flatten the indexing, but
       *      Bloc is a two-dimensional array 
       *      (see definitiona at the begining of kernels).
       **************************************************************/

      // Please implement Task 3.1.2 here

      __syncthreads();

      // compute the per-thread result css:
      for(int k = 0; k < Tk; k++) {
          #pragma unroll
          for(int i=0; i<Ry; i++) {
              #pragma unroll
              for(int j=0; j<Rx; j++) {
                /***************************************
                 * Cuda Exercise 3:
                 * Task 3.2:
                 * Please modify the code statement below to
                 * refer to Aloc and Bloc instead of A and B.
                 * This assumes of course that you have 
                 *   already solved Task 3.1.
                 ***************************************/
                  css[i][j] +=  
                    A[ (iii + threadIdx.y*Rx + i)*widthA + (kk + k)] *
                    B[ (kk+k)*widthB + jjj + threadIdx.y*Ry + j] ;
              }
          }
      }
      __syncthreads();
  }

  unsigned int indy = iii + threadIdx.y * Ry;
  unsigned int indx = jjj + threadIdx.x * Rx;

  // Update C in global memory with the per-thread result css.
  #pragma unroll
  for(int i=0; i<Ry; i++) {
    #pragma unroll
    for(int j=0; j<Rx; j++) {
      if( (indy+i < heightA) && (indx+j < widthB) )
        C[(indy+i)*widthB + (indx+j)] = css[i][j];
    }
  }
}

#endif
