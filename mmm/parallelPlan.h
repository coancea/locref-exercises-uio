#ifndef CUDA_LIKE_SEQ
#define CUDA_LIKE_SEQ

#define min(a,b) ( ((a)<(b))? (a) : (b) )

/**
 * Computes matrix multiplication C = A*B
 * Semantically the matrix sizes are:
 *    A : [heightA][widthA]ElTp
 *    B : [ widthA][widthB]ElTp
 *    C : [heightA][widthB]ElTp
 *  for some numeric type ElTp.
 **/
template<class ElTp, int T, int R>
void cudaLikeSeq(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  const int TR = T*R;

  #pragma omp parallel for schedule(static) collapse(2)
  for(int iii=0; iii<heightA; iii+=TR) {  // parallel (on Grid.y)
    for(int jjj=0; jjj<widthB; jjj+=TR) { // parallel (on Grid.x)
      for(int ii=iii; ii<min(iii+TR,heightA); ii+=R) {  // parallel (on Block.y)
        for(int jj=jjj; jj<min(jjj+TR,widthB); jj+=R) { // parallel (on Block.x)
          ElTp css[R][R];

          // initialize css
          for(int q=0; q<R; q++)
            for(int p=0; p<R; p++)
              css[q][p] = 0;

          for(int kk=0; kk<widthA; kk+=T) {
            // We remap the slices of A and B which would have
            //   been read in the remaining code in this scope
            //   to smaller buffers Aloc and Bloc. The slices are:
            //   A[ii : ii+R][kk : kk+T] and
            //   B[kk : kk+T][jj : jj+R]
            ElTp Aloc[R][T];
            ElTp Bloc[T][R];
            for(int i_r=0; i_r<R; i_r++)
              for(int k_r=0; k_r<T; k_r++) {
                const int i = ii+i_r, k = kk + k_r;
                Aloc[i_r][k_r] = (i < heightA && k < widthA) ?
                                 A[i*widthA + k] : 0;
              }

            for(int k_r=0; k_r<T; k_r++)
              for(int j_r=0; j_r<R; j_r++) {
                const int j = jj+j_r, k = kk + k_r;
                Bloc[k_r][j_r] = (j < widthB && k < widthA) ?
                                 B[k*widthB + j] : 0;
              }
            // However, in Cuda, we would have to consider the slices
            //   of A and B read by the entire Cuda-block (i.e., parallel
            //   dimensions ii and jj) and collectively copy them to shared
            //   memory with all the threads in the block. The slices are:
            //     A[iii : iii+T*R][kk : kk+T] and
            //     B[kk : kk+T][jjj : jjj+T*R]
            //   Since the Cuda block has dimension TxT it follows that
            //     a thread will remap R elements from A and R from B.            

            // MAIN COMPUTATION (with Remapped A and B)
            #pragma unroll
            for(int k_r=0; k_r<T; k_r++) { // original k = kk + k_r
              #pragma unroll
              for(int i_r=0; i_r<R; i_r++) { // original i = ii + i_r
                #pragma unroll
                for(int j_r=0; j_r<R; j_r++) { // original j = jj + j_r
                  css[i_r][j_r] += Aloc[i_r][k_r] * Bloc[k_r][j_r];
                    //with the original arrays it would be
                    // A[(ii+i_r)*widthA + (kk+k_r)] * B[(kk+k_r)*widthB + (jj+j_r)];

                } // end loop j_r
              } // end loop i_r
            } // end loop k_r
          } // end loop kk

          // update the global C with the per-thread results css
          for(int i_r=0; i_r<R; i_r++) {
            const int i = ii+i_r;

            for(int j_r=0; j_r<R; j_r++) {
              const int j = jj+j_r;
              if(i < heightA && j < widthB)
                C[i*widthB + j] = css[i_r][j_r];
            } // end loop j_r
          } // end loop i_r

        } // end loop jj
      } // end loop ii
    } // end loop jjj
  } // end loop iii
}

#endif
