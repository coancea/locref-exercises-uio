#include "../helper.h"
#include "goldenSeq.h"
#include "kernels.cu.h"

#define GPU_RUNS    30
#define ERR         0.000001

#define LLC_FRAC (3.0 / 7.0)

#define DEVICE_INFO  0

template<int B>
void multiStepHisto ( uint32_t* d_inp_inds
                    , float*    d_inp_vals
                    , float*    d_hist
                    , const uint32_t N
                    , const uint32_t H
                    , const uint32_t LLC
) {
    // we use a fraction L of the last-level cache (LLC) to hold `hist`
    const uint32_t CHUNK = ( LLC_FRAC * LLC ) / sizeof(float);
    uint32_t num_partitions = (H + CHUNK - 1) / CHUNK;

    //printf( "Number of partitions: %f\n", ((float)H)/CHUNK );

    uint32_t grid = (N + B - 1) / B;
    cudaMemset(d_hist, 0, H * sizeof(float));

    /************************
     *** Cuda Exercise 1: ***
     ************************
     * 1. Please introduce the chunking loop of count `num_partitions`
     *    around the kernel call below, similar to parallelPlan.h.
     * 2. Then set the correct lower/upper bounds as the last two
     *    parameters of the kernel call, such as to implement the
     *    multi-step traversal that processes indices falling only
     *    in the current chunk.
     * 3. Then modify the multiStepKernel code in kernels.cu.h
     ****************************************************************/
    {
        multiStepKernel<<<grid,B>>>(d_inp_inds, d_inp_vals, d_hist, N, 0, H);
    }
}

/**
 * N is the length of the input;
 * H is the length of the histogram;
 * LLC is supposed to be the size of the LLC cache in bytes
 */
template<int B>
void runAll ( const uint32_t N, const uint32_t H, const uint32_t LLC ) {
    // set seed for rand()
    srand(2006);
 
    // 1. allocate memory
    uint32_t* inp_inds  = (uint32_t*) malloc( ((size_t)N) * sizeof(uint32_t) );
    float*    inp_vals  = (float*) malloc( ((size_t)N) * sizeof(float) );
    float*    hist_gold = (float*) malloc( ((size_t)H) * sizeof(float) );
    float*    h_hist    = (float*) malloc( ((size_t)H) * sizeof(float) );
 
    // 2. initialize arrays
    randomInit<float>(inp_vals, N);
    randomInds(inp_inds, N, H);

    for(uint32_t i=0; i<H; i++) {
        hist_gold[i] = 0;
        h_hist[i] = 0;
    }

    // get reference result
    goldenSeq(inp_inds, inp_vals, hist_gold, N, H);

    // 3. allocate device memory
    uint32_t* d_inp_inds;
    float*    d_inp_vals;
    float*    d_hist;
    
    cudaMalloc((void**) &d_inp_inds, N * sizeof(uint32_t) );
    cudaMalloc((void**) &d_inp_vals, N * sizeof(float) );
    cudaMalloc((void**) &d_hist, H * sizeof(float) );
 
    // 4. initialize arrays on device
    cudaMemcpy(d_inp_inds, inp_inds, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inp_vals, inp_vals, N * sizeof(float),    cudaMemcpyHostToDevice);


    // 5. get cuda properties 
    uint32_t HWD; // max num of hwd threads
    {
        int nDevices;
        cudaGetDeviceCount(&nDevices);

        cudaDeviceProp prop;

        cudaGetDeviceProperties(&prop, 0);
        HWD = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
        const uint32_t BLOCK_SZ = prop.maxThreadsPerBlock;
        const uint32_t SH_MEM_SZ = prop.sharedMemPerBlock;
        if (DEVICE_INFO) {
            printf("Device name: %s\n", prop.name);
            printf("Number of hardware threads: %d\n", HWD);
            printf("Block size: %d\n", BLOCK_SZ);
            printf("Shared memory size: %d\n", SH_MEM_SZ);
            puts("====");
        }
    }

    // 5. declaration of timing structures
    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    { // 5. naive GPU version
        uint32_t grid = (N + B - 1) / B;

        // dry run
        cudaMemset(d_hist, 0, H * sizeof(float));
        naiveKernel<<<grid,B>>>(d_inp_inds, d_inp_vals, d_hist, N, H);
        cudaDeviceSynchronize();
    
        // measure average of runs
        gettimeofday(&t_start, NULL); 

        for (int kkk = 0; kkk < GPU_RUNS; kkk++) {
            cudaMemset(d_hist, 0, H * sizeof(float));
            naiveKernel<<<grid,B>>>(d_inp_inds, d_inp_vals, d_hist, N, H);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;
        gigaBytesPerSec = (16.0 * N + 4.0 * H) * 1.0e-3f / elapsed;
        printf("Original Histogram runs on GPU in: %lu microsecs, GB/sec: %f\n"
              , elapsed, gigaBytesPerSec);

        gpuAssert( cudaPeekAtLastError() );

        // copy result from device to host
        cudaMemcpy(h_hist, d_hist, H * sizeof(float), cudaMemcpyDeviceToHost);

        validate<float>(h_hist, hist_gold, H, ERR);
    }

    { // 5. Multi-Step Traversal to Optimize LLC Threshing
        // dry run
        multiStepHisto<B>(d_inp_inds, d_inp_vals, d_hist, N, H, LLC);
        cudaDeviceSynchronize();
    
        // measure average of runs
        gettimeofday(&t_start, NULL); 

        for (int kkk = 0; kkk < GPU_RUNS; kkk++) {
            multiStepHisto<B>(d_inp_inds, d_inp_vals, d_hist, N, H, LLC);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;
        gigaBytesPerSec = (16.0 * N + 4.0 * H) * 1.0e-3f / elapsed;
        printf("Multi-Step Histogram runs on GPU in: %lu microsecs, GB/sec: %f\n"
              , elapsed, gigaBytesPerSec);

        gpuAssert( cudaPeekAtLastError() );

        // copy result from device to host
        cudaMemcpy(h_hist, d_hist, H * sizeof(float), cudaMemcpyDeviceToHost);

        validate<float>(h_hist, hist_gold, H, ERR);
    }


    // clean up memory
    free(inp_inds);
    free(inp_vals);
    free(hist_gold);
    free(h_hist);
    cudaFree(d_inp_inds);
    cudaFree(d_inp_vals);
    cudaFree(d_hist);
}

int main (int argc, char * argv[]) {

    if (argc != 3) {
        printf("Usage: %s <N> <LLC>\n", argv[0]);
        exit(1);
    }
    const uint32_t N  = atol(argv[1]);
    const uint32_t LLC = atol(argv[2]);
    
    // we set the histogram size, such that to take 4 passes,
    // where we assume a fraction LLC_FRAC of the last-level
    // cache (LLC) is dedicated to storing the histogram
    const uint32_t H = ( LLC_FRAC * LLC ) - 10;

    printf("Running GPU-Parallel Versions (Cuda) demonstrating LLC threshing on histogram example. N: %d, H: %d, LLC: %d\n"
          , N, H, LLC);

    runAll<256>( N, H, LLC);
    return 0;
}
