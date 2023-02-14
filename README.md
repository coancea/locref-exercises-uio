# Demonstrating Locality of Reference on Multi-Cores and GPU

This repository consists of several code examples that are intended to demonstrate the importance of optimizing the locality of reference, specialized in the context of GPU hardware, but also multi-core CPUs.

Four Cuda exercises are provided that require you to fill-in-the-blanks some key (missing) parts of implementation, without which the code either does not validate or has su-optimal performance. Please find details about exercises in the `README.md` of each of their corresponding folders.

## histo-L3-thrashing

Illustrates a multi-step technique for eliminating threshing of the last-level cache (LLC) for histogram-like computations [1] at the expense of redundant computation. The idea is to partition the histogram (such that a partition fits in LLC), and, in each sequential step, to process in parallel only the indices that fall inside the current partition. For GPU this results in significant speedup (e.g., 3x), while for multi-core CPU the speedup seems to be modest.

[1] Troels Henriksen, Sune Hellfritzsch, Ponnuswamy Sadayappan, Cosmin Oancea, "Compiling Generalized Histograms for GPU", SC'20, [link](https://futhark-lang.org/publications/sc20.pdf)

## gpu-coalescing

Illustrates a rather general technique for improving spatial locality in the context of the GPU hardware --- i.e., ensuring coalesced access to global memory --- by means of transposing the input array(s) and computing the result array in transposed form. This strategy provides significant speedups (e.g., 9x on some GPUs) in comparison with the naive version that uses uncoalesced accesses, even though it requires three times more memory accesses.   On multi-core CPU this strategy results in significant slowdowns, because in this example, good spatial locality on GPU means exactly the opposite that on CPUs (i.e., coalesced access on GPU means that 16 consecutive threads access during the same SIMD load/store instruction consecutive elements in global memory; this would be terrible for CPU due to coherency false-sharing misses). 

## mmm-sols

Contains several implementations for matrix multiplication:

- CPU (multi-core) variants include: (1) naive OMP version, (2) an OMP version that exhibits optimal spatial locality by transposing the second array, and (3) a Cuda-like version that is parallelized with OMP that uses register and block tiling (targeting reuse from L1 and registers).

- GPU variants include: (1) a naive one, and (2) another one that uses block and register tiling --- the read arrays are held in shared memory and the result array is computed in register memory.

## batch-mmm-sols

Implements a batched matrix multiplication computational kernel, in which the same two matrices are multiplied under a mask (representing missing data), but the accumulation happens only when the corresponding element from the mask is true (valid data). This is a simplified version from a real-world computational kernel used in analyses of satellite images to detect landscape changes (BFAST) [2].

Multi-core CPU and GPU variants include a straightforwardly parallelized version (naive), and another one in which the outermost parallel dimension is strip-mined, and the corresponding tile is sequentialized and interchanged to the innermost position. This strategy enables reuse from registers (but for optimal GPU results it also requires to first transpose the mask, and then to copy slices of it to shared memory in coalesced fashion).

[2] Fabian Geseke, Sabina Rosca, Troels Henriksen, Jan Verbesselt and Cosmin Oancea, "Massively-Parallel Change Detection for Satellite Time Series Data with Missing Values", ICDE’20, [link](https://futhark-lang.org/publications/icde20.pdf)

## Code Structure of Each Exercise/Folder:

```goldenSeq.h``` contains the so called golden sequential version, i.e., the simplest implementation. The name is misleading because the code is also parallelized by means of OpenMP for multi-core execution. The performance of the other implementations is measured in GBytes/sec or Gflops/sec, where the number of accessed bytes or the number of flops is computed according to the golden sequential specification.

```parallelPlan.h``` contains an OpenMP-parallelized version that is as close as you can get in C to the optimized Cuda version of that exercise/example. Interestingly, in the case of (batched) matrix multiplication, this version also achieve significant speedups on multi-core CPUs.

```main-omp.c``` contains the glue code for running (and validating) the OpenMP versions and reporting the performance.

```kernels.cu.h``` contains the implementation of the Cuda kernels.

```main-gpu.cu``` contains the glue code for running (and validating) the Cuda versions and reporting the performance.

Located at root level, ```helper.h``` contains common functionality, used by multiple exercises.

## How to Compile/Run

The repository assumes that a standard installation of Cuda and of ```g++``` with support for OpenMP (```-fopenmp```) are available.

- ```make``` will compile and run both the OpenMP and Cuda versions.

- ```make run_gpu``` and ```make run_cpu``` will compile and run the Cuda (GPU) versions and the OpenMP (multi-core) versions, respectively.

- ```make compile_gpu``` and ```make compile_cpu``` will only compile the Cuda (GPU) versions and the OpenMP (multi-core) versions, respectively.

- The number of threads used by OpenMP can be set in the corresponding terminal, for example with a command similar to:

```
$ export OMP_NUM_THREADS=8
```

## Setting Up Environment Variables

After you log in on one of `ml1-7.hpc.uio.no`
you would need to add the following to your `.bashrc` file, 
then log out and log back in again (for the settings to take effect).

```
CUDA_DIR=/storage/software/CUDA/11.7.0
PATH=$CUDA_DIR/bin:$PATH
LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH
CPLUS_INCLUDE_PATH=$CUDA_DIR/include:$CPLUS_INCLUDE_PATH
C_INCLUDE_PATH=$CUDA_DIR/include:$C_INCLUDE_PATH
CPATH=$CUDA_DIR/include:$CPATH
LIBRARY_PATH=$CUDA_DIR/lib64:$LIBRARY_PATH

export LD_LIBRARY_PATH
export PATH
export CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH
export CPATH
export LIBRARY_PATH
```


