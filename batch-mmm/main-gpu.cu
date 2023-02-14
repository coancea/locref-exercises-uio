#include "../helper.h"
#include "kernels.cu.h"
#include "goldenSeq.h"
#include "parallelPlan.h"

using namespace std;

#define GPU_RUNS    200
#define ERR         0.0000012

/**
 * Naive kernel: the only tiling performed is on the grid;
 *               no shared or private memory is used.
 * d_A, d_B, d_X are the input matrices stored on the device;
 * ref_Y is the reference-result matrix computed with goldenSeq
 *       and stored on the host.
 * d_Y is the result array, which is to be computed on the
 *     device and validated against ref_Y.
 * For matrix sizes, please see goldenSeq.
 * T is the generic (numeric) array-element type.
 * TL is the block size on the X and Y dimensions and
 * TZ is the block size on the Z dimension.
 */
__host__
void runNaive ( float* d_A,   float* d_B, char*  d_X
              , float* ref_Y, float* d_Y
              , const int M, const int K, const int N
) {
    unsigned long long size_Y = M*K*K;
    unsigned long long mem_size_Y = size_Y * sizeof(float);
    cudaMemset (d_Y, 0, mem_size_Y );

    // setup execution parameters
    dim3 block(K, K, 1);
    dim3 grid (M, 1, 1);

    // dry run
    bmmmNaiveKer<float> <<< grid, block >>>(d_A, d_B, d_X, d_Y, M, K, N);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
    
    for(int i=0; i<GPU_RUNS; i++) {
        bmmmNaiveKer<float> <<< grid, block >>>(d_A, d_B, d_X, d_Y, M, K, N);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    float microsecPerMatrixMul = elapsed; 
    double flopsPerMatrixMul = 3.0 * M * K * K * N; 
    double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerMatrixMul; 

    printf("GPU Naive BMMM version runs in: %lu microsecs, GFlops/sec: %.2f\n", elapsed, gigaFlops);

    float* h_Y = (float*) malloc(mem_size_Y);
    cudaMemcpy(h_Y, d_Y, mem_size_Y, cudaMemcpyDeviceToHost);
    validate<float>(ref_Y, h_Y, size_Y, ERR);
    free(h_Y);
}

template<typename T, int TILE>
void runCudaTranspose(T* d_A, T* d_A_tr, const int heightA, const int widthA) {
    const int dimy = (heightA + TILE - 1) / TILE;
    const int dimx = (widthA  + TILE - 1) / TILE;
    dim3 block(TILE, TILE, 1);
    dim3 grid (dimx, dimy, 1);
    matTransposeTiledKer<T, 32><<<grid, block>>>(d_A, d_A_tr, heightA, widthA);
}

template<int TR> __host__
void runTiled( float* d_A, float* d_B
              , char* d_X, char* d_X_tr
              , float* ref_Y, float* d_Y
              , const int M, const int K
              , const int N
) {
    unsigned long long size_Y = M*K*K;
    unsigned long long mem_size_Y = size_Y * sizeof(float);
    cudaMemset (d_Y, 0, mem_size_Y );

    // setup execution parameters
    const int  dim_grid = (M + TR- 1) / TR;

    dim3 block(K, K, 1);
    dim3 grid (dim_grid, 1, 1);

    // dry run
    runCudaTranspose<char,32>(d_X, d_X_tr, M, N);
    bmmmTiledKer<float,TR><<< grid, block >>>(d_A, d_B, d_X_tr, d_Y, M, K, N);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
    
    for(int i=0; i<GPU_RUNS; i++) {
        runCudaTranspose<char, 32>(d_X, d_X_tr, M, N);
        bmmmTiledKer<float,TR><<< grid, block >>>(d_A, d_B, d_X_tr, d_Y, M, K, N);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    double  microsecPerMatrixMul = elapsed;
    double flopsPerMatrixMul = 3.0 * M * K * K * N;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerMatrixMul; 

    gpuAssert( cudaPeekAtLastError() );
    printf("GPU RegTiled BMMM simple version runs in: %.1f microsecs, GFlops/sec: %.2f\n", elapsed, gigaFlops);

    float* h_Y = (float*) malloc(mem_size_Y);
    cudaMemcpy(h_Y, d_Y, mem_size_Y, cudaMemcpyDeviceToHost);
    validate<float>(ref_Y, h_Y, size_Y, ERR);
    free(h_Y);
}

/**
 * This will run all code versions
 * (and summarize the performance in GFlops). 
 */
template<int TR>
void runAll ( const int M, const int K, const int N ) {
    srand(2022);
 
    // 1. allocate host memory for the four matrices A, B, X, Y
    unsigned long long size_A = K * N;
    unsigned long long mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
 
    unsigned long long size_B = N * K;
    unsigned long long mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    unsigned long long size_X = M * N;
    unsigned long long mem_size_X = sizeof(char) * size_X;
    char* h_X = (char*) malloc(mem_size_X);

    unsigned long long size_Y = M * K * K;
    unsigned long long mem_size_Y = sizeof(float) * size_Y;
    float* h_Y = (float*) malloc(mem_size_Y);
 
    // 2. initialize input arrays in host memory
    randomInit<float>(h_A, size_A);
    randomInit<float>(h_B, size_B);
    randomMask(h_X, size_X, 0.1);
    
    // 3. allocate device memory
    float *d_A, *d_B, *d_Y;
    char *d_X, *d_X_tr;
    cudaMalloc((void**) &d_A,    mem_size_A);
    cudaMalloc((void**) &d_B,    mem_size_B);
    cudaMalloc((void**) &d_X,    mem_size_X);
    cudaMalloc((void**) &d_X_tr, mem_size_X);
    cudaMalloc((void**) &d_Y,    mem_size_Y);
 
    // 4. copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, mem_size_X, cudaMemcpyHostToDevice);
  
    printf("Sizes are: (M, K, N)=(%d, %d, %d)\n", M, K, N);

    // 5. compute golden sequential
    {
        goldenSeq(h_A, h_B, h_X, h_Y, M, K, N);
    }

    // 6. compute cuda-like sequential implementation
    {
        float* h_Y1  = (float*)malloc(mem_size_Y);

        cudaLikeSeq<8>( h_A, h_B, h_X, h_Y1, M, K, N );

        printf("Cuda-like sequential version: ");

        validate<float>(h_Y, h_Y1, size_Y, ERR);

        free(h_Y1);
    }
    // 6. compute the naive GPU version
    runNaive(d_A, d_B, d_X, h_Y, d_Y, M, K, N);

    // 8. compute the simple register-tiled version (blockDim.z == 1)
    runTiled<TR>(d_A, d_B, d_X, d_X_tr, h_Y, d_Y, M, K, N);

    free(h_A);
    free(h_B);
    free(h_X);
    free(h_Y);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_X);
    cudaFree(d_X_tr);
    cudaFree(d_Y);
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main (int argc, char * argv[]) {
    if (argc != 4) {
        printf("Usage: %s M K N\n", argv[0]);
        exit(1);
    }
    printf("Running GPU-Parallel Versions (Cuda) of Batch-MMM\n");
    const int M = atoi(argv[1]);
    const int K = atoi(argv[2]);
    const int N = atoi(argv[3]);

    runAll<31> ( M, K, N );
}
