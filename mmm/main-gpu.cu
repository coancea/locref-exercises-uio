#include "../helper.h"
#include "goldenSeq.h"
#include "kernels.cu.h"

using namespace std;

#define GPU_RUNS    100
#define ERR         0.000005

// naive kernel, i.e., the only tiling performed is on the grid;
//   no shared or private memory is used.
template< typename T, int TL>
__host__ void runNaive(  int height_A, int width_A, int width_B,
                T* d_A, T* d_B, T* d_C, T* h_C
             ) {

    unsigned long long mem_size_C = height_A*width_B*sizeof(T);
    cudaMemset (d_C, 0, mem_size_C );

    // setup execution parameters
    int  dimy = (height_A + TL - 1) / TL; 
    int  dimx = (width_B  + TL - 1) / TL;

    dim3 block(TL, TL, 1);
    dim3 grid (dimx, dimy, 1);

    // dry run
    mmmNaiveKer<T> <<< grid, block >>>(d_A, d_B, d_C, height_A, width_B, width_A);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
    
    for(int i=0; i<GPU_RUNS; i++) {
        mmmNaiveKer<T> <<< grid, block >>>(d_A, d_B, d_C, height_A, width_B, width_A);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    float microsecPerMatrixMul = elapsed; 
    double flopsPerMatrixMul = 2.0 * height_A * width_B * width_A; 
    double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerMatrixMul; 

    printf("GPU Naive MMM version runs in: %lu microsecs, GFlops/sec: %.2f\n", elapsed, gigaFlops);

    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);    
}

//  symmetric block-and-register tiling with inner dimension sequential
template<class T, int Ty, int Ry, int Tx, int Rx, int Tk>
void runSymetricBlkRegTileInnSeq( 
        int HEIGHT_A, int WIDTH_A, int WIDTH_B,
        T* d_A, T* d_B, T* d_C, T* ref_C
    ) {

    unsigned long long size_C = HEIGHT_A * WIDTH_B;
    unsigned long long mem_size_C = size_C * sizeof(T);
    T* h_C = (T*) malloc(mem_size_C);

    // setup execution parameters
    int  dimy = ceil( ((float)HEIGHT_A)/(Ty*Ry) ); 
    int  dimx = ceil( ((float) WIDTH_B)/(Tx*Rx) );
    dim3 block(Tx, Ty, 1);
    dim3 grid (dimx, dimy, 1);

    { // one dry run
        mmmSymBlkRegInnSeqKer<T,Ty,Ry,Tx,Rx,Tk> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A); 
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    }

    cudaMemset(d_C, 0, mem_size_C);

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
      
    for(int i=0; i<GPU_RUNS; i++) {
        mmmSymBlkRegInnSeqKer<T,Ty,Ry,Tx,Rx,Tk> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

    float microsecPerMatrixMul = elapsed; 
    double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
    double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerMatrixMul; 

    printf( "GPU Blk-Reg Tiled MMM version runs in: %lu microsecs, GFlops/sec: %.2f\n"
          , elapsed, gigaFlops );


    // copy result from device to host and validate
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    validate<T>(ref_C, h_C, size_C, ERR);
    free(h_C);
}

template<class T, int TL, int REG>
void runAll ( int height_A, int width_A, int width_B ) {

    srand(2006);
 
    // 1. allocate host memory for the two matrices
    unsigned long long size_A = width_A * height_A;
    unsigned long long mem_size_A = sizeof(T) * size_A;
    T* h_A = (T*) malloc(mem_size_A);
 
    unsigned long long size_B = width_B * width_A;
    unsigned long long mem_size_B = sizeof(T) * size_B;
    T* h_B = (T*) malloc(mem_size_B);
 
    // 2. initialize host memory
    randomInit<T>(h_A, size_A);
    randomInit<T>(h_B, size_B);
    
    // 3. allocate device memory
    T* d_A;
    T* d_B;
    cudaMalloc((void**) &d_A, mem_size_A);
    cudaMalloc((void**) &d_B, mem_size_B);
 
    // 4. copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
 
    // 5. allocate host memory for the result C
    unsigned int size_C = height_A * width_B;
    unsigned int mem_size_C = sizeof(T) * size_C;
    //T* h_C   = (T*) malloc(mem_size_C);
    T* ref_C = (T*) malloc(mem_size_C);
 
    // 6. allocate device memory for the result
    T *d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    printf("Sizes are: (HeightA, WidthB, WidthA)=(%d, %d, %d)\n", height_A, width_B, width_A);

    runNaive<T, TL>( height_A, width_A, width_B, d_A, d_B, d_C, ref_C );

    runSymetricBlkRegTileInnSeq<T, TL, REG, TL, REG, TL>( height_A, width_A, width_B, d_A, d_B, d_C, ref_C );

    free(h_A);
    free(h_B);
    free(ref_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main (int argc, char * argv[]) {
    if (argc != 4) {
        printf("Usage: %s heiht-A width-A width-B\n", argv[0]);
        exit(1);
    }
    const int HEIGHT_A = atoi(argv[1]);
    const int WIDTH_A  = atoi(argv[2]);
    const int WIDTH_B  = atoi(argv[3]);

    printf("Running GPU-Parallel Versions (Cuda) of MMM\n");

    runAll<float, 16, 5> ( HEIGHT_A, WIDTH_A, WIDTH_B );
}
