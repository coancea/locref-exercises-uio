using namespace std;

#define cudaError_t int
#define cudaSuccess 1
#define cudaGetErrorString(a) "Phony"

#include "../helper.h"
#include "goldenSeq.h"
#include "parallelPlan.h"

#define CPU_RUNS    30
#define ERR         0.0000012

int main (int argc, char * argv[]) {
    if (argc != 4) {
        printf("Usage: %s M K N\n", argv[0]);
        exit(1);
    }
    const int M = atoi(argv[1]);
    const int K = atoi(argv[2]);
    const int N = atoi(argv[3]);

    printf("Running CPU-Parallel Versions (OMP) of Batch-MMM\n");

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

// 5. compute golden sequential
    {
        // dry run
        goldenSeq(h_A, h_B, h_X, h_Y, M, K, N);

        double elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        for(int i=0; i<CPU_RUNS; i++)
            goldenSeq(h_A, h_B, h_X, h_Y, M, K, N);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / CPU_RUNS;
        double flopsPerMatrixMul = 3.0 * M * K * K * N;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / elapsed; 

        printf("Golden sequential OMP version runs in: %.1f microsecs, GFlops/sec: %.2f\n", elapsed, gigaFlops);
    }

    // 6. compute cuda-like sequential implementation
    {
        float* h_Y1  = (float*)malloc(mem_size_Y);

        // dry run
        cudaLikeSeq<4>( h_A, h_B, h_X, h_Y1, M, K, N );

        double elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        for(int i=0; i<CPU_RUNS; i++)
            cudaLikeSeq<4>( h_A, h_B, h_X, h_Y1, M, K, N );

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / CPU_RUNS;
        double flopsPerMatrixMul = 3.0 * M * K * K * N;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / elapsed; 

        printf("Cuda-like OMP version runs in: %.1f microsecs, GFlops/sec: %.2f\n", elapsed, gigaFlops);

        validate<float>(h_Y, h_Y1, size_Y, ERR);

        free(h_Y1);
    }
}
