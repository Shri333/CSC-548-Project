// implementation of odd-even sort
// based on "A Version of Parallel Odd-Even Sorting Algorithm Implemented in CUDA Paradigm" by Ajdari et al.
// author: Shrihan Dadi (sdadi2)
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "common.cuh"
using namespace std;

void usage() {
    cout << "usage: odd-even [k]" << endl;
    cout << "where 2^k is the size of the vector to generate for sorting" << endl;
    exit(1);
}

// special case if vec is small enough: odd-even sort vec in one block
__global__ void smallOddEvenSort(float vec[], size_t size) {
    // copy vec into shared memory
    __shared__ float sharedVec[NUM_THREADS];
    sharedVec[threadIdx.x] = vec[threadIdx.x];
    __syncthreads();

    // n - 1 iterations, alternating odd and even swaps
    for (size_t i = 0; i < size - 1; i++) {
        if (threadIdx.x < size / 2 && i % 2 == 0) { // even
            cmpSwap(sharedVec, 2 * threadIdx.x, 2 * threadIdx.x + 1);
        } else if (threadIdx.x < size / 2 && 2 * threadIdx.x + 2 < size) { // odd
            cmpSwap(sharedVec, 2 * threadIdx.x + 1, 2 * threadIdx.x + 2);
        }
        __syncthreads();
    }

    // copy shared memory back into vec (after sorting with odd-even sort)
    vec[threadIdx.x] = sharedVec[threadIdx.x];
}

// sort a partition of the global vector on each block using bitonic sort
__global__ void localBitonicSort(float vec[], size_t size) {
    size_t idx = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    // copy partition of vector into shared memory
    __shared__ float sharedVec[NUM_THREADS / 2];
    sharedVec[threadIdx.x] = vec[idx];
    __syncthreads();

    // bitonic sort this sub-vector/partition
    for (unsigned int phase = 1; phase <= 9; phase++) {
        for (unsigned int step = phase; step >= 1; step--) {
            if (threadIdx.x < NUM_THREADS / 4) {
                bitonicSwap(sharedVec, NUM_THREADS / 2, phase, step, threadIdx.x);
            }
            __syncthreads();
        }
    }

    // copy sorted sub-vector back into vec
    vec[idx] = sharedVec[threadIdx.x];
}

// bitonic merge two sorted subarrays/partitions into one subarray/partition
__global__ void bitonicMerge(float vec[], size_t size, bool even) {
    size_t idx = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx >= size || (!even && idx + NUM_THREADS >= size)) {
        return;
    }

    // copy 2 sorted partitions of vector into shared memory
    __shared__ float sharedVec[NUM_THREADS];
    if (even) {
        sharedVec[threadIdx.x] = vec[idx];
    } else {
        sharedVec[threadIdx.x] = vec[idx + NUM_THREADS / 2];
    }
    __syncthreads();

    // merge two sorted partitions
    for (unsigned int phase = 1; phase <= 10; phase++) {
        for (unsigned int step = phase; step >= 1; step--) {
            if (threadIdx.x < NUM_THREADS / 2) {
                bitonicSwap(sharedVec, NUM_THREADS, phase, step, threadIdx.x);
            }
            __syncthreads();
        }
    }

    // copy sorted partition in shared memory back into vector
    if (even) {
        vec[idx] = sharedVec[threadIdx.x];
    } else {
        vec[idx + NUM_THREADS / 2] = sharedVec[threadIdx.x];
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        usage();
    }

    // read k from argv[1] where 2^k is the size of the vector to generate
    istringstream ss(argv[1]);
    unsigned int k;
    if (!(ss >> k) || k > sizeof(size_t) * 8 - 1) {
        usage();
    }

    // generate vector
    size_t size = 1 << k;
    thrust::host_vector<float> vec = genVec(size);

    // sort with normalized bitonic sort
    cout << "Sorting vector of size " << size << "..." << endl;
    thrust::device_vector<float> gpuVec = vec;
    float* gpuVecPtr = thrust::raw_pointer_cast(gpuVec.data());

    // time sorting
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    if (size <= NUM_THREADS) {
        // special case: we can sort the entire vector in one block
        smallOddEvenSort<<<1, size>>>(gpuVecPtr, size);
    } else {
        // sort a partition of the vector on each block w/ bitonic sort
        size_t numBlocks = size / (NUM_THREADS / 2);
        localBitonicSort<<<numBlocks, NUM_THREADS / 2>>>(gpuVecPtr, size);

        // size / (NUM_THREADS / 2) - 1 iterations: alternating odd/even bitonic merges
        numBlocks = size / NUM_THREADS;
        size_t iterations = size / (NUM_THREADS / 2) - 1;
        for (size_t i = 0; i < iterations; i++) {
            bitonicMerge<<<numBlocks, NUM_THREADS>>>(gpuVecPtr, size, i % 2 == 0);
        }
    }
    cudaEventRecord(stop);

    // copy gpuVec back into vec
    vec = gpuVec;

    // get time to sort
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // print out time to sort
    cout << "Time: " << milliseconds << " ms" << endl;

#ifdef DEBUG
    if (!sorted(vec)) {
        cout << "vec is not sorted!" << endl;
    }
#endif

    return 0;
}
