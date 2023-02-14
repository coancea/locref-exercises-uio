using namespace std;

#define cudaError_t int
#define cudaSuccess 1
#define cudaGetErrorString(a) "Phony"

#include "../helper.h"
#include "goldenSeq.h"
#include "spatial-only-opt.h"
#include "parallelPlan.h"

#define CPU_RUNS_GOLD  4
#define CPU_RUNS    (CPU_RUNS_GOLD * 8)
#define ERR         0.0000012

template<class T, int TL, int REG>
void runAll ( int height_A, int width_A, int width_B ) {
    srand(2006);
 
    // 1. allocate memory for the three matrices
    unsigned long long size_A = width_A * height_A;
    unsigned long long mem_size_A = sizeof(T) * size_A;
    T* A = (T*) malloc(mem_size_A);
 
    unsigned long long size_B = width_B * width_A;
    unsigned long long mem_size_B = sizeof(T) * size_B;
    T* B = (T*) malloc(mem_size_B);

    unsigned int size_C = height_A * width_B;
    unsigned int mem_size_C = sizeof(T) * size_C;
    T* C_gold = (T*) malloc(mem_size_C);
    T* C_spopt= (T*) malloc(mem_size_C);
    T* C_opt  = (T*) malloc(mem_size_C);
 
    // 2. initialize host memory
    randomInit<T>(A, size_A);
    randomInit<T>(B, size_B);

    // 3. compute golden sequential
    {
        //dry run
        goldenSeq<T>(A, B, C_gold, height_A, width_B, width_A);

        double elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        for(int i=0; i<CPU_RUNS_GOLD; i++)
            goldenSeq<T>(A, B, C_gold, height_A, width_B, width_A);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / CPU_RUNS_GOLD;
        double flopsPerMatrixMul = 2.0 * height_A * width_B * width_A;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / elapsed; 

        printf("Golden sequential OMP version of MMM runs in: %.1f microsecs, GFlops/sec: %.2f\n", elapsed, gigaFlops);
    }

    // 4. compute spatial-optimized by transposition
    {
        spatialOpt<T>( A, B, C_spopt, height_A, width_B, width_A);

        double elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);
 
        for(int i=0; i<CPU_RUNS; i++)
            spatialOpt<T>( A, B, C_spopt, height_A, width_B, width_A);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / CPU_RUNS;
        double flopsPerMatrixMul = 2.0 * height_A * width_B * width_A;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / elapsed; 

        printf("Spatially optimised OMP version of MMM runs in: %.1f microsecs, GFlops/sec: %.2f\n"
              , elapsed, gigaFlops);

        validate<T>(C_gold, C_spopt, size_C, ERR);
    }

    // 5. compute cuda-like sequential implementation (block and register tiled)
    {
        // dry run
        cudaLikeSeq<T,TL,REG>( A, B, C_opt, height_A, width_B, width_A);

        double elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);
 
        for(int i=0; i<CPU_RUNS; i++)
            cudaLikeSeq<T,TL,REG>( A, B, C_opt, height_A, width_B, width_A);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / CPU_RUNS;
        double flopsPerMatrixMul = 2.0 * height_A * width_B * width_A;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / elapsed; 

        printf("Cuda-like OMP version of MMM runs in: %.1f microsecs, GFlops/sec: %.2f\n", elapsed, gigaFlops);

        validate<T>(C_gold, C_opt, size_C, ERR);
    }

    free(A);
    free(B);
    free(C_gold);
    free(C_spopt);
    free(C_opt);
}

int main (int argc, char * argv[]) {
    if (argc != 4) {
        printf("Usage: %s heiht-A width-A width-B\n", argv[0]);
        exit(1);
    }
    const int HEIGHT_A = atoi(argv[1]);
    const int WIDTH_A  = atoi(argv[2]);
    const int WIDTH_B  = atoi(argv[3]);

    printf("Running CPU-Parallel Versions (OMP) of MMM\n");

    runAll<float, 4, 32> ( HEIGHT_A, WIDTH_A, WIDTH_B );
}
