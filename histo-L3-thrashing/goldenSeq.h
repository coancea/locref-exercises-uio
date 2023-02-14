#ifndef GOLDEN
#define GOLDEN

/**
 * Input:
 *   inp_inds : [N]int
 *   inp_vals : [N]float
 * Result:
 *   hist : [H]float
 * Assumption:
 *   inp_inds holds indices greater or equal to 0 and less than H;
 */
void goldenSeq( uint32_t* inp_inds
              , float*    inp_vals
              , float*    hist
              , const uint32_t N
              , const uint32_t H
) {
    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < N; i++) {
        uint32_t ind = inp_inds[i];
        float    val = inp_vals[i];

        if(ind < H) { // sanity, should hold.
            #pragma omp atomic
            hist[ind] += val;
        }
    }
}

#endif
