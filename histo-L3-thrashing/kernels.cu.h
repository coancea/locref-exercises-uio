#ifndef HIST_KERNELS
#define HIST_KERNELS

__global__ void
naiveKernel ( uint32_t* inp_inds
            , float*    inp_vals
            , volatile float* hist
            , const uint32_t N
            , const uint32_t H
) {
    uint32_t gid = blockIdx.x*blockDim.x + threadIdx.x;
  
    if(gid < N) {
        uint32_t ind = inp_inds[gid];
        float    val = inp_vals[gid];
        if(ind < H) {
            atomicAdd((float*)&hist[ind], val);
        }
    }
}

__global__ void
multiStepKernel ( uint32_t* inp_inds
                , float*    inp_vals
                , volatile float* hist
                , const uint32_t N
                // the lower & upper bounds
                // of the current chunk
                , const uint32_t LB
                , const uint32_t UB
) {
    const uint32_t gid = blockIdx.x*blockDim.x + threadIdx.x;

    if(gid < N) {
        uint32_t ind = inp_inds[gid];

        /************************
         *** Cuda Exercise 1: ***
         * 
         * Change the if condition to succeed
         * only when the current index `ind` is
         * within the bounds of the current chunk
         * (less than UB and greater or equal to LB.)
         ************************/

        if(ind < H) {
            float val = inp_vals[gid];
            atomicAdd((float*)&hist[ind], val);
        }
    }
}

#endif
