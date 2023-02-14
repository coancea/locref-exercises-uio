#ifndef HELPER
#define HELPER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#if 0
typedef int        int32_t;
typedef long long  int64_t;
#endif

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define min(a,b) ( ((a)<(b))? (a) : (b) )

template<typename T, int TILE>
void runSeqTranspose(T* A, T* A_tr, const int heightA, const int widthA) {
  #pragma omp parallel for collapse(2)
  for(int ii=0; ii<heightA; ii+=TILE) {
    for(int jj=0; jj<widthA; jj+=TILE) {

        for(int i=ii; i<min(ii+TILE,heightA); i++) {
          for(int j=jj; j<min(jj+TILE,widthA); j++) {
            A_tr[j*heightA + i] = A[i*widthA + j];
          }
        }
      
    }
  }
}

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

template<class T>
void randomInit(T* data, uint64_t size) {
    for (uint64_t i = 0; i < size; i++)
        data[i] = rand() / (T)RAND_MAX;
}

template<class T>
void zeroInit(T* data, uint64_t size) {
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < size; i++)
        data[i] = 0;
}

void randomInds(uint32_t* data, uint64_t size, uint32_t M) {
    for (uint64_t i = 0; i < size; i++)
        data[i] = rand() % M;
}

/**
 * Initialize the `data` array, which has `size` elements:
 * frac% of them are NaNs and (1-frac)% are random values.
 * 
 */
void randomMask(char* data, uint64_t size, float frac) {
    for (uint64_t i = 0; i < size; i++) {
        float r = rand() / (float)RAND_MAX;
        data[i] = (r >= frac) ? 1 : 0;
    }
}

// error for matmul: 0.02
template<class T>
bool validate(T* A, T* B, const uint64_t sizeAB, const T ERR){
    for(uint64_t i = 0; i < sizeAB; i++) {
        T curr_err = fabs( (A[i] - B[i]) / fmax(A[i], B[i]) ); 
        if (curr_err >= ERR) {
            printf("INVALID RESULT at flat index %llu: %f vs %f\n", i, A[i], B[i]);
            return false;
        }
    }
    printf("VALID RESULT!\n");
    return true;
}

template<class T>
bool validateExact(T* A, T* B, uint64_t sizeAB){
    for(uint64_t i = 0; i < sizeAB; i++) {
        if ( A[i] != B[i] ) {
            printf("INVALID RESULT at flat index %llu: %f vs %f\n", i, (float)A[i], (float)B[i]);
            return false;
        }
    }
    printf("VALID RESULT!\n");
    return true;
}

#endif
