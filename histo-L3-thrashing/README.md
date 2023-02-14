# Histogram-like Computation

See section "LL$ threshing: Histogram-like computation" in companion lecture slides.

The programming task refers to implementing the missing code in files `main-gpu.cu` and `kernels.cu.h`---search for keyword "Exercise" in those files and follow the instructions.

- The first argument of the program is the size `N` of the array of indices/values. 

- The second argument of the program is the size of the last-level cache (LL$) in bytes. Please make sure to adjust it to the hardware you are running on (both CPU and GPU), otherwise you will not observe much.

- The size of the histogram is computed internally such as four passes over the input are always performed.

