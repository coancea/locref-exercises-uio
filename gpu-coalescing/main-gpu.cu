#include "../helper.h"
#include "goldenSeq.h"
#include "kernels.cu.h"

#define GPU_RUNS    100
#define ERR         0.000001

template<class ElTp>
int validateTranspose(ElTp* A, ElTp* trA, const uint32_t rowsA, const uint32_t colsA){
  int valid = 1;
  for(uint64_t i = 0; i < rowsA; i++) {
    for(uint64_t j = 0; j < colsA; j++) {
      if(trA[j*rowsA + i] != A[i*colsA + j]) {
        printf("row: %llu, col: %llu, A: %.4f, trA: %.4f\n"
              , i, j, A[i*colsA + j], trA[j*rowsA + i] );
        valid = 0;
        break;
      }
    }
    if(!valid) break;
  }
  if (valid) printf("GPU TRANSPOSITION   VALID!\n");
  else       printf("GPU TRANSPOSITION INVALID!\n");
  return valid;
}

/**
 * Input:
 *   inp_d : [height][width]ElTp
 * Result:
 *   out_d : [width][height]ElTp
 *   (the transpose of inp_d.)
 */
template<class ElTp, int T>
void callTransposeKer( ElTp*          inp_d,  
                       ElTp*          out_d, 
                       const uint32_t height, 
                       const uint32_t width,
                       const uint32_t with_coalesced
) {
    // 1. setup block and grid parameters
    int  dimy = (height+T-1) / T; 
    int  dimx = (width +T-1) / T;
    dim3 block(T, T, 1);
    dim3 grid (dimx, dimy, 1);

    //2. execute the kernel
    if(with_coalesced)
        coalsTransposeKer<ElTp,T> <<< grid, block >>>
                        (inp_d, out_d, height, width);
    else 
        naiveTransposeKer<ElTp> <<< grid, block >>>
                        (inp_d, out_d, height, width);
}

template<class ElTp, int B> void
runTranspose( ElTp* d_A, ElTp* d_B
            , ElTp* h_A, ElTp* h_B
            , const uint32_t num_rows
            , const uint32_t num_cols
            , const uint32_t with_coalesced
) {
    uint64_t mem_size = num_rows * num_cols * sizeof(ElTp);
    // 1. dry run
    callTransposeKer<ElTp, B>( d_A, d_B, num_rows, num_cols, with_coalesced );

    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 

    // 2. measure the average of runs
    for(int i=0; i<GPU_RUNS; i++)
        callTransposeKer<ElTp, B>( d_A, d_B, num_rows, num_cols, with_coalesced );
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;
    gigaBytesPerSec = 2 * mem_size * 1.0e-3f / elapsed;
    printf("Transpose on GPU with coalesced = %d runs in: %lu microsecs, GB/sec: %f\n"
          , with_coalesced, elapsed, gigaBytesPerSec);

    gpuAssert( cudaPeekAtLastError() );

    // 3. copy result from device to host
    cudaMemcpy(h_B, d_B, mem_size, cudaMemcpyDeviceToHost);

    // 4. validate
    validateTranspose<ElTp>( h_A, h_B, num_rows, num_cols );
}


template<class ElTp, int B>
void runAll ( const uint32_t num_rows, const uint32_t num_cols ) {
    // set seed for rand()
    srand(2006);
 
    // 1. allocate host memory for the two matrices
    size_t size = num_rows * num_cols;
    size_t mem_size = sizeof(ElTp) * size;
    ElTp* h_A = (ElTp*) malloc(mem_size);
    ElTp* h_B = (ElTp*) malloc(mem_size);
    ElTp* h_B_ref = (ElTp*) malloc(mem_size);
 
    // 2. initialize host memory
    randomInit<ElTp>(h_A, size);
    
    // 3. allocate device memory
    ElTp* d_A;
    ElTp* d_B;
    cudaMalloc((void**) &d_A, mem_size);
    cudaMalloc((void**) &d_B, mem_size);
 
    // 4. copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);

#if 1
    // Input:  d_A : [num_rows][num_cols]ElTp
    // Result: d_B : [num_cols][num_rows]ElTp, the transpose of d_A.
    runTranspose<ElTp,32>(d_A, d_B, h_A, h_B, num_rows, num_cols, 0);
    runTranspose<ElTp,32>(d_A, d_B, h_A, h_B, num_rows, num_cols, 1);
#endif

    goldenSeq(h_A, h_B_ref, num_rows, num_cols);

    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    { // 5. naive GPU version
        uint32_t grid = (num_rows + B - 1) / B;

        // dry run
        cudaMemset(d_B, 0, mem_size);
        naiveKernel<<<grid,B>>>(d_A, d_B, num_rows, num_cols);
        cudaMemset(d_B, 0, mem_size);
        cudaDeviceSynchronize();
    
        // measure average of runs
        gettimeofday(&t_start, NULL); 

        for (int kkk = 0; kkk < GPU_RUNS; kkk++) {
            naiveKernel<<<grid,B>>>(d_A, d_B, num_rows, num_cols);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;
        gigaBytesPerSec = 2 * mem_size * 1.0e-3f / elapsed;
        printf("Original Program runs on GPU in: %lu microsecs, GB/sec:%f\n"
              , elapsed, gigaBytesPerSec);

        gpuAssert( cudaPeekAtLastError() );

        // copy result from device to host
        cudaMemcpy(h_B, d_B, mem_size, cudaMemcpyDeviceToHost);

        validate<ElTp>(h_B, h_B_ref, size, ERR);
    }

    { // 6. Program with coalesced accesses optimized by independent 
      //      transpositions, i.e., which are manifested in memory.
      //    This will show a significant speed-up w.r.t. the naive
      //      version even if it performs 3x more memory accesses
      //      (due to having optimal spatial locality, i.e., coalescing).

        uint32_t grid = (num_rows + B - 1) / B;

        ElTp* d_Atr;
        ElTp* d_Btr;
        cudaMalloc((void**) &d_Btr, mem_size);
        cudaMalloc((void**) &d_Atr, mem_size);

        // dry run
        cudaMemset(d_B, 0, mem_size);
        callTransposeKer<ElTp, 32>( d_A,  d_Atr, num_rows, num_cols, 1 );
        transKernel<<<grid, B>>>(d_Atr, d_Btr, num_rows, num_cols );
        callTransposeKer<ElTp, 32>(d_Btr, d_B,   num_cols, num_rows, 1 );
        cudaMemset(d_B, 0, mem_size);
        cudaDeviceSynchronize();

        // measure average of runs
        gettimeofday(&t_start, NULL); 

        for (int kkk = 0; kkk < GPU_RUNS; kkk++) {
            callTransposeKer<ElTp, 32>( d_A,  d_Atr, num_rows, num_cols, true );
            transKernel<<<grid, B>>>(d_Atr, d_Btr, num_rows, num_cols );
            callTransposeKer<ElTp, 32>(d_Btr, d_B,   num_cols, num_rows, true );
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;
        gigaBytesPerSec = 2 * mem_size * 1.0e-3f / elapsed;
        printf("Coalesced Program by transpositions runs on GPU in: %lu microsecs, GB/sec: %f\n"
              , elapsed, gigaBytesPerSec);

        gpuAssert( cudaPeekAtLastError() );

        // copy result from device to host and validate
        cudaMemcpy(h_B, d_B, mem_size, cudaMemcpyDeviceToHost);
        validate<ElTp>(h_B, h_B_ref, size, ERR);

        cudaFree(d_Atr);
        cudaFree(d_Btr);
   }

    { // 7. optimized program---i.e., exhibiting only coalesced 
      //    accesses---obtained by using the shared memory as
      //    a staging buffer, i.e., read from global-to-shared
      //    memory (in coalesced way) and then each thread reads
      //    from shared memory in non-coalesced way. Note that
      //    this version should be the fastest, as it does not
      //    require to perform the transpositions.

        const int CHUNK = 16;
        if((B % CHUNK) != 0) {
            printf("Broken Assumption: block size not a multiple of chunk size, EXITING!\n");
            exit(1);
        }
        uint32_t grid = (num_rows + B - 1) / B;

        // dry run
        cudaMemset(d_B, 0, mem_size);
        optimKernel<ElTp,CHUNK><<<grid, B, CHUNK*B*sizeof(ElTp)>>>(d_A, d_B, num_rows, num_cols);
        cudaDeviceSynchronize();
        cudaMemset(d_B, 0, mem_size);

        // measure average of runs
        gettimeofday(&t_start, NULL); 

        for (int kkk = 0; kkk < GPU_RUNS; kkk++) {
            optimKernel<ElTp,CHUNK><<<grid, B, CHUNK*B*sizeof(ElTp)>>>(d_A, d_B, num_rows, num_cols);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 
        gigaBytesPerSec = 2 * mem_size * 1.0e-3f / elapsed;
        printf("Optimal Program by shared-mem-staging runs on GPU in: %lu microsecs, GB/sec: %f\n"
              , elapsed, gigaBytesPerSec);

        gpuAssert( cudaPeekAtLastError() );

        // copy result from device to host and validate
        cudaMemcpy(h_B, d_B, mem_size, cudaMemcpyDeviceToHost);
        validate<ElTp>(h_B, h_B_ref, size, ERR);
    }


   // clean up memory
   free(h_A);
   free(h_B);
   free(h_B_ref);
   cudaFree(d_A);
   cudaFree(d_B);
}

int main(int argc, char * argv[]) {
    if (argc != 2) {
        printf("Usage: %s <number-of-rows>\n", argv[0]);
        exit(1);
    }
    const uint32_t num_rows = atoi(argv[1]);
    const uint32_t num_cols = 64;

    printf("Running GPU-Parallel Versions (Cuda) demonstrating (un)coalesced accesses to global memory\n");

    runAll<float, 256>( num_rows, num_cols );
    return 0;
}

