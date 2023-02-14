#ifndef GOLDEN
#define GOLDEN

/**
 * Input:
 *   A : [num_rows][num_cols]ElTp
 * Result:
 *   B : [num_rows][num_cols]ElTp
 */
template<class ElTp>
void goldenSeq(ElTp* A, ElTp* B, const uint32_t num_rows, const uint32_t num_cols) {
    #pragma omp parallel for schedule(static) 
    for(uint64_t i = 0; i < num_rows; i++) {
        uint64_t ii = i*num_cols;
        ElTp accum = 0.0;
        for(uint64_t j = 0; j < num_cols; j++) {
            ElTp a_el  = A[ii + j];
            accum = sqrt(accum) + a_el*a_el;
            B[ii + j] = accum;
        }
    }
}

#endif
