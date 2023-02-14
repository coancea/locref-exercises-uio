using namespace std;

#define cudaError_t int
#define cudaSuccess 1
#define cudaGetErrorString(a) "Phony"

#include "../helper.h"
#include "goldenSeq.h"
#include "parallelPlan.h"

#define CPU_RUNS    5
#define ERR         0.000001

void initZero(float* hist, const uint32_t H) {
    for(uint32_t i=0; i<H; i++) {
        hist[i] = 0;
    }
}


/**
 * N is the length of the input;
 * H is the length of the histogram;
 * L3 is supposed to be the size of the L3 cache in bytes
 */
void runAll ( const uint32_t N, const uint32_t H, const uint32_t L3 ) {
    // set seed for rand()
    srand(2006);
 
    // 1. allocate memory
    uint32_t* inp_inds  = (uint32_t*) malloc( ((size_t)N) * sizeof(uint32_t) );
    float*    inp_vals  = (float*) malloc( ((size_t)N) * sizeof(float) );
    float*    hist_gold = (float*) malloc( ((size_t)H) * sizeof(float) );
    float*    hist_optm = (float*) malloc( ((size_t)H) * sizeof(float) );
 
    // 2. initialize arrays
    randomInit<float>(inp_vals, N);
    randomInds(inp_inds, N, H);
    
    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    // 3. golden sequential version, parallelized by OMP.
    //    this should thresh the L3 cache if H is high enough.
    {
        // dry run
        initZero(hist_gold, H);
        goldenSeq(inp_inds, inp_vals, hist_gold, N, H);
    
        // measure average of runs
        gettimeofday(&t_start, NULL); 

        for (int kkk = 0; kkk < CPU_RUNS; kkk++) {
            initZero(hist_gold, H);
            goldenSeq(inp_inds, inp_vals, hist_gold, N, H);
        }

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / CPU_RUNS;
        gigaBytesPerSec = (16.0 * N + 4.0 * H) * 1.0e-3f / elapsed;
        printf("Original Histogram runs on CPU in: %lu microsecs, GB/sec: %f\n"
              , elapsed, gigaBytesPerSec);
    }

    // 4. Cuda-like implementation that uses the multi-pass technique
    //    to redundantly reaverse the dataset several time, such that
    //    each pass performs the updates of a partition of the histogram
    //    that fits in the L3 cache.
    {
        // dry run
        initZero(hist_optm, H);
        cudaLikeSeq(inp_inds, inp_vals, hist_optm, N, H, L3);
    
        // measure average of runs
        gettimeofday(&t_start, NULL); 

        for (int kkk = 0; kkk < CPU_RUNS; kkk++) {
            initZero(hist_optm, H);
            cudaLikeSeq(inp_inds, inp_vals, hist_optm, N, H, L3);
        }

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / CPU_RUNS;
        gigaBytesPerSec = (16.0 * N + 4.0 * H) * 1.0e-3f / elapsed;
        printf("Multi-Pass Histogram runs on CPU in: %lu microsecs, GB/sec: %f\n"
              , elapsed, gigaBytesPerSec);

        validate<float>(hist_optm, hist_gold, H, ERR);
    }

    // clean up memory
    free(inp_inds);
    free(inp_vals);
    free(hist_gold);
    free(hist_optm);
}

int main (int argc, char * argv[]) {

    if (argc != 3) {
        printf("Usage: %s <N> <L3>\n", argv[0]);
        exit(1);
    }
    const uint32_t N  = atol(argv[1]);
    const uint32_t L3 = atol(argv[2]);

    //const uint32_t H  = atol(argv[2]);
    // we set the histogram size, such that to take 4 passes,
    // where we assume 4/7 of the L3 cache is used by the histogram
    const uint32_t H = 4 * (L3 / 7) - 10;

    printf("Running CPU-Parallel Versions (OMP) demonstrating L3 threshing on histogram example. N: %d, H: %d, L3: %d\n"
          , N, H, L3);

    runAll( N, H, L3);
    return 0;
}
