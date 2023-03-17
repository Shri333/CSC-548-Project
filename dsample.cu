// implementation of deterministic sample sort
// based on "Deterministic Sample Sort For GPUs" by Dehne and Zaboli
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
    cout << "usage: dsample [k]" << endl;
    cout << "where 2^k is the size of the vector to generate for sorting" << endl;
    exit(1);
}

// special case if vec's size is less than NUM_THREADS
__global__ void smallBitonicSort(float vec[], unsigned int k) {
    size_t size = 1 << k;
    for (unsigned int phase = 1; phase <= k; phase++) {
        for (unsigned int step = phase; step >= 1; step--) {
            bitonicSwap(vec, size, phase, step, threadIdx.x);
            __syncthreads();
        }
    }
}

// sort a partition of the global vector on each block using bitonic sort
__global__ void localBitonicSort(float vec[], size_t size) {
    size_t idx = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;

    // copy partition of vector into shared memory
    __shared__ float sharedVec[NUM_THREADS];
    sharedVec[threadIdx.x] = vec[idx];
    __syncthreads();

    // bitonic sort this sub-vector/partition
    for (unsigned int phase = 1; phase <= 10; phase++) {
        for (unsigned int step = phase; step >= 1; step--) {
            if (threadIdx.x < NUM_THREADS / 2) {
                bitonicSwap(sharedVec, NUM_THREADS, phase, step, threadIdx.x);
            }
            __syncthreads();
        }
    }

    // copy sorted sub-vector back into vec
    vec[idx] = sharedVec[threadIdx.x];
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
        smallBitonicSort<<<1, size / 2>>>(gpuVecPtr, k);
    } else {
        // sort a partition of the vector on each block w/ bitonic sort
        size_t numBlocks = size / NUM_THREADS;
        localBitonicSort<<<numBlocks, NUM_THREADS>>>(gpuVecPtr, size);
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
