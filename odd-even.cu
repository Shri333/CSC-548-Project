// implementation of odd-even sort
// based on "A Version of Parallel Odd-Even Sorting Algorithm Implemented in CUDA Paradigm" by Ajdari et al.
// author: Shrihan Dadi (sdadi2)
#include <iostream>
#include <sstream>
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
    sharedVec[2 * threadIdx.x] = vec[2 * threadIdx.x];
    sharedVec[2 * threadIdx.x + 1] = vec[2 * threadIdx.x + 1];
    __syncthreads();

    // n - 1 iterations, alternating odd and even swaps
    for (size_t i = 0; i < size - 1; i++) {
        if (i & 1 == 0) { // even
            cmpSwap(vec, 2 * threadIdx.x, 2 * threadIdx.x + 1);
        } else if (2 * threadIdx.x + 2 < size) { // odd
            cmpSwap(vec, 2 * threadIdx.x + 1, 2 * threadIdx.x + 2);
        }
        __syncthreads();
    }

    // copy shared memory back into vec (after sorting with odd-even sort)
    vec[2 * threadIdx.x] = sharedVec[2 * threadIdx.x];
    vec[2 * threadIdx.x + 1] = sharedVec[2 * threadIdx.x + 1];
}

// sort a partition of the global vector on each block using bitonic sort
__global__ void localBitonicSort(float vec[], size_t size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size)
        return;

    // copy partition of vector into shared memory
    __shared__ float subVec[NUM_THREADS / 2];
    subVec[threadIdx.x] = vec[idx];
    __syncthreads();

    // bitonic sort this sub-vector/partition
    for (unsigned int phase = 1; phase <= 9; phase++) {
        for (unsigned int step = phase; step >= 1; step--) {
            bitonicSwap(subVec, NUM_THREADS / 2, phase, step, threadIdx.x);
            __syncthreads();
        }
    }

    // copy sorted sub-vector back into vec
    vec[idx] = subVec[threadIdx.x];
}

int main(int argc, char** argv) {
    if (argc != 2)
        usage();

    // read k from argv[1] where 2^k is the size of the vector to generate
    istringstream ss(argv[1]);
    unsigned int k;
    if (!(ss >> k) || k > sizeof(size_t) * 8 - 1)
        usage();

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
        smallOddEvenSort<<<1, size / 2>>>(gpuVecPtr, size);
    } else {
        size_t numBlocks = size / (NUM_THREADS / 2)
        if (size % (NUM_THREADS / 2) != 0)
            numBlocks++;

        // sort a partition of the vector on each block w/ bitonic sort
        localBitonicSort<<<numBlocks, NUM_THREADS / 2>>>(gpuVecPtr, size);

        numBlocks = size / NUM_THREADS;
        if (size % NUM_THREADS == 0)
            numBlocks++;

        // 
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
    if (!sorted(vec))
        cout << "vec is not sorted!" << endl;
#endif

    return 0;
}
