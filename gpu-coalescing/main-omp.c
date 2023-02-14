using namespace std;

#define cudaError_t int
#define cudaSuccess 1
#define cudaGetErrorString(a) "Phony"

#include "../helper.h"
#include "goldenSeq.h"
#include "parallelPlan.h"

#define CPU_RUNS    10
#define ERR         0.0000001


template<class ElTp>
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

    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    // 3. golden sequential version, parallelized by OMP.
    //    this should give good performance, as it has
    //    perfect spatial locality on CPU
    //    (but un-coalesced on GPUs)/
    {
        // dry run
        goldenSeq(h_A, h_B_ref, num_rows, num_cols);
    
        // measure average of runs
        gettimeofday(&t_start, NULL); 

        for (int kkk = 0; kkk < CPU_RUNS; kkk++) {
                goldenSeq(h_A, h_B_ref, num_rows, num_cols);
        }

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / CPU_RUNS;
        gigaBytesPerSec = 2 * mem_size * 1.0e-3f / elapsed;
        printf("Original Program runs on CPU in: %lu microsecs, GB/sec: %f\n"
              , elapsed, gigaBytesPerSec);
    }

    // 4. Cuda-like implementation that would achieves coalesced
    //    access on GPUs, but has very poor spatial locality on CPUs.
    {
        // dry run
        cudaLikeSeq(h_A, h_B, num_rows, num_cols);
    
        // measure average of runs
        gettimeofday(&t_start, NULL); 

        for (int kkk = 0; kkk < CPU_RUNS; kkk++) {
                cudaLikeSeq(h_A, h_B, num_rows, num_cols);
        }

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / CPU_RUNS;
        gigaBytesPerSec = 2 * mem_size * 1.0e-3f / elapsed;
        printf("Program with GPU-like coalescing runs on CPU in: %lu microsecs, GB/sec: %f\n"
              , elapsed, gigaBytesPerSec);

        validate<ElTp>(h_B, h_B_ref, size, ERR);
    }

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_B_ref);
}

int main (int argc, char * argv[]) {

    if (argc != 2) {
        printf("Usage: %s <number-of-rows>\n", argv[0]);
        exit(1);
    }
    const uint32_t num_rows = atoi(argv[1]);
    const uint32_t num_cols = 64;

    printf("Running CPU-Parallel Versions (OMP) demonstrating (un)coalesced accesses to global memory\n");

    runAll<float>( num_rows, num_cols );
    return 0;
}
