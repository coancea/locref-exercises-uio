#ifndef CUDA_LIKE_SEQ
#define CUDA_LIKE_SEQ

/**
 * Input:
 *   inp_inds : [N]int
 *   inp_vals : [N]float
 * Result:
 *   hist : [H]float
 * Assumption:
 *   inp_inds holds indices greater or equal to 0 and less than H;
 * Motivation/Intuition:
 *   If `H` is large than `hist` will not fit in the L3 cache,
 *     and the atomic updates to `hist` will result in threshing
 *     (the L3 cache).
 *   We use a simple optimization strategy: we perform a (smallish)
 *     number of traversals of the input, each traversal updating only
 *     the indices that fall into a partition of `hist`, which is small
 *     enough to fit in the L3 cache.
 */
void
cudaLikeSeq( uint32_t* inp_inds
           , float*    inp_vals
           , float*    hist
           , const uint32_t N
           , const uint32_t H
           , const uint32_t L3
) {
    // we use 4/5 of the L3 cache to hold `hist`
    const uint32_t CHUNK = ( 4 * (L3 / 7) ) / sizeof(float);
    uint32_t num_partitions = (H + CHUNK - 1) / CHUNK;

    //printf( "Number of partitions: %f\n", ((float)H)/CHUNK );

    for(uint32_t k=0; k<num_partitions; k++) {
        // we process only the indices falling in
        // the integral interval [k*CHUNK, (k+1)*CHUNK)
        uint32_t low_bound = k*CHUNK;
        uint32_t upp_bound = min( (k+1)*CHUNK, H );

        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < N; i++) {
            uint32_t ind = inp_inds[i];

            if(ind >= low_bound && ind < upp_bound) {
                float    val = inp_vals[i];

                #pragma omp atomic
                hist[ind] += val;
            }
        }
    }
}

#endif
