// header for common code used by all parallel sorting algorithms
// author: Shrihan Dadi (sdadi2)
#include <thrust/host_vector.h>
#include <cstddef>

#pragma once

// max number of threads per block in CUDA
#define NUM_THREADS 1024

// function for generating a random vector of floats
thrust::host_vector<float> genVec(size_t size);

// compares the values of the elements at the two indices (i and j)
// and swaps if vec[i] > vec[j]
__device__ void cmpSwap(float vec[], size_t i, size_t j);

// function for performing a normalized bitonic swap
// given a step and phase and thread index
__device__ void bitonicSwap(float vec[], size_t size, unsigned int phase, unsigned int step, size_t idx);

// function for checking if a vector of floats is sorted or not
// only used in DEBUG mode
#ifdef DEBUG
    bool sorted(const thrust::host_vector<float>& vec);
#endif
