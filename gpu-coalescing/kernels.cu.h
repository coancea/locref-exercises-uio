#ifndef TRANSPOSE_KERS
#define TRANSPOSE_KERS

typedef unsigned int uint32_t; 

// widthA = heightB
template <class T> 
__global__ void
naiveTransposeKer(T* A, T* B, int heightA, int widthA) {

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  if( (gidx >= widthA) || (gidy >= heightA) ) return;

  B[gidx*heightA+gidy] = A[gidy*widthA + gidx];
}

// blockDim.y = T; blockDim.x = T
// each block transposes a square T
template <class ElTp, int T> 
__global__ void
coalsTransposeKer(ElTp* A, ElTp* B, int heightA, int widthA) {
  __shared__ ElTp tile[T][T+1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if( x < widthA && y < heightA )
      tile[threadIdx.y][threadIdx.x] = A[y*widthA + x];

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x; 
  y = blockIdx.x * T + threadIdx.y;

  if( x < heightA && y < widthA )
      B[y*heightA + x] = tile[threadIdx.x][threadIdx.y];
}

/**
 * Input:
 *   A : [num_rows][num_cols]ElTp
 * Result:
 *   B : [num_rows][num_cols]ElTp
 * Observations:
 *   1. This is the naive version because one can observe
 *        that all accesses to A and B are un-coalesced.
 *   2. The dimension of size `num_rows` is parallel, the
 *        one of size `num_cols` is sequential. 
 */
template<class ElTp>
__global__ void 
naiveKernel(ElTp* A, ElTp* B, uint32_t num_rows, uint32_t num_cols) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= num_rows) return;

    ElTp accum = 0;
    for(int j=0; j<num_cols; j++) {
        ElTp el_a  = A[ gid*num_cols + j ];
        accum = sqrt(accum) + el_a * el_a;
        B[ gid*num_cols + j ] = accum;
    }
}


/**
 * ElTp: some numeric type, e.g., float.
 * Input:
 *   A_tr : [num_cols][num_rows]ElTp
 * Result:
 *   B_tr : [num_cols][num_rows]ElTp
 *
 *************************************
 * Cuda Exercise 2:
 **********************
 * Please implement the optimized kernel below, which
 * is supposed to use the transposed version of A (named A_tr)
 * and to compute the transposed version of B (named B_tr),
 * in order to achieve fully-coalesced access to global
 * memory (for matrices A_tr, B_tr). 
 * 
 * Please note that `num_rows` and `num_cols` refer to the
 * number of rows and columns of the original matrices A and B.
 * 
 * Remember that:
 *  1. the dimension of size `num_rows` is parallel, 
 *     and has been mapped across Cuda threads; 
 *  2. the dimension of size `num_cols` is sequential and
 *     should appear as a loop in the kernel code you implement
 *      (similar to kernel naiveKernel above).
 *  3. you need to flatten the indexing, i.e., A_tr and B_tr
 *      are one-dimensional arrays.
 **************************************************************/
template<class ElTp>
__global__ void 
transKernel(ElTp* A_tr, ElTp* B_tr, uint32_t num_rows, uint32_t num_cols) {
}

///////////////////////////////////
///////////////////////////////////
///////////////////////////////////

template<int CHUNK>
__device__ inline void 
glb2shmem( uint32_t num_cols
         , uint32_t block_offs
         , uint32_t jj
         , uint32_t j
         , uint32_t chunk_lane
         , uint32_t chunk_id // input
         , uint32_t* glb_ind
         , uint32_t* loc_ind // the result  
) {
    *loc_ind = j*blockDim.x + threadIdx.x;
    *glb_ind = block_offs + j*(blockDim.x/CHUNK)*num_cols + chunk_id*num_cols + jj + chunk_lane;
}

template<class ElTp, int CHUNK>
__global__ void 
optimKernel(ElTp* A, ElTp* B, uint32_t num_rows, uint32_t num_cols) { //unsigned int N) {
    extern __shared__ ElTp sh_mem[]; // length: CHUNK * blockDim.x
    
    uint32_t block_offs = blockIdx.x * blockDim.x * num_cols;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    ElTp accum = 0.0;

    uint32_t chunk_lane = threadIdx.x % CHUNK;
    uint32_t chunk_id   = threadIdx.x / CHUNK;

    for(int jj=0; jj<num_cols; jj+=CHUNK) {
        uint32_t count_j = min(jj+CHUNK, num_cols);
       
        // look at shared memory as a "[CHUNK][blockDim.x]ElTp" array
        // each thread in the current block copies CHUNK elements in
        // coalesced way from global-to-shared memory
        for(int j=0; j<CHUNK; j++) {
            uint32_t loc_ind, glb_ind;
            glb2shmem<CHUNK>( num_cols, block_offs, jj, j, chunk_lane, chunk_id, &glb_ind, &loc_ind );
            if(glb_ind < num_rows*num_cols) {
                sh_mem[loc_ind] = A[glb_ind];
            } 
        }
        __syncthreads();

        if(gid < num_rows)
        for(int j=jj; j<count_j; j++) {
            // look at shared memory as a "[blockDim.x][CHUNK]ElTp" array
            // each thread copies its CHUNK consecutive elements from
            // shared memory and updates in-place shared memory with its
            // result
            uint32_t loc_ind = threadIdx.x * CHUNK + j-jj;
            ElTp tmpA = sh_mem[loc_ind];
            accum = sqrt(accum) + tmpA*tmpA;
            sh_mem[loc_ind] = accum;
        }
        __syncthreads();

        // look at shared memory as a "[CHUNK][blockDim.x]ElTp" array
        // threads cooperatively write in coalesced form the result
        // from shared-to-global memory (each thread writes CHUNK elements)
        for(int j=0; j<CHUNK; j++) {
            uint32_t loc_ind, glb_ind;
            glb2shmem<CHUNK>( num_cols, block_offs, jj, j, chunk_lane, chunk_id, &glb_ind, &loc_ind );
            if(glb_ind < num_rows*num_cols) {
                B[glb_ind] = sh_mem[loc_ind];
            } 
        }
        __syncthreads();
    }
}

#endif
