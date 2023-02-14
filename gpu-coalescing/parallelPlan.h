#ifndef CUDA_LIKE_SEQ
#define CUDA_LIKE_SEQ

/**
 * Input:
 *   A : [num_rows][num_cols]ElTp
 * Result:
 *   B : [num_rows][num_cols]ElTp
 * Implementation:
 *   This is the Cuda-like CPU implementation that ensures
 *     coalesced access to global memory if the hardware
 *     would be GPU---i.e., 16 consecutive threads access
 *     16 consecutive memory locations. 
 *   The implementation transposes the input array, and
 *     computes the result array in transposed form; then
 *     transposes the (transposed) result.
 *   This is quite the oposite of what good spatial locality
 *     means on CPU, so we expect that this version will have
 *     abysmal performance on multicore CPUs, e.g., due to
 *     frequent false-sharing cache conflicts.
 **/

template<class ElTp>
void cudaLikeSeq( ElTp* A
                , ElTp* B
                , const uint32_t num_rows
                , const uint32_t num_cols
) {
    // allocate memory for the transposed arrays
    uint64_t mem_size = sizeof(ElTp) * num_rows * num_cols;
    ElTp* A_tr = (ElTp*) malloc(mem_size);
    ElTp* B_tr = (ElTp*) malloc(mem_size);

    // transpose input
    runSeqTranspose<ElTp,64>(A, A_tr, num_rows, num_cols);
    
    // computation too slow to run in parallel (de-comment if you insist)
    #pragma omp parallel for schedule(dynamic) 
    for(uint64_t i = 0; i < num_rows; i++) { // parallel
        ElTp accum = 0.0;
        for(uint64_t j = 0; j < num_cols; j++) { // sequential
            ElTp a_el  = A_tr[j*num_rows + i];
            accum = sqrt(accum) + a_el*a_el;
            B_tr[j*num_rows + i] = accum;
        }
    }

    runSeqTranspose<ElTp,64>(B_tr, B, num_cols, num_rows);
}

#endif
