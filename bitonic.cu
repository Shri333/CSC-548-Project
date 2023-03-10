// implementation of bitonic sort
// based on "Fast in-place, comparison-based sorting with CUDA: a study with bitonic sort" by Peters et al.
// author: Shrihan Dadi (sdadi2)
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "common.h"
using namespace std;

void usage() {
    cout << "usage: bitonic [k]" << endl;
    cout << "where 2^k is the size of the vector to generate for sorting" << endl;
    exit(1);
}

// compares the values of the elements at the two indices (i and j)
// and swaps if gpuVec[i] > gpuVec[j]
__device__ void cmpSwap(float* gpuVecPtr, size_t i, size_t j) {
    if (gpuVecPtr[i] > gpuVecPtr[j]) {
        float tmp = gpuVecPtr[i];
        gpuVecPtr[i] = gpuVecPtr[j];
        gpuVecPtr[j] = tmp;
    }
}

// kernel for bitonic sort
__global__ void bitonicSort(float* gpuVecPtr, size_t gpuVecSize, unsigned int phase, unsigned int step) {
    size_t idx = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx >= gpuVecSize / 2)
        return;
    size_t groupSize = 1 << step;
    size_t threadsPerGroup = groupSize / 2;
    size_t groupIdx = idx / threadsPerGroup;
    size_t i = groupIdx * groupSize + (idx % threadsPerGroup);
    size_t j;
    if (step == phase) {
        // first step: normalized swap
        j = (groupSize * (groupIdx + 1) - 1) - (idx % threadsPerGroup);
    } else {
        // rest of the steps: repeated bitonic merge
        j = i + threadsPerGroup;
    }
    cmpSwap(gpuVecPtr, i, j);
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
    size_t numBlocks = size / NUM_THREADS;
    if (size % NUM_THREADS != 0)
        numBlocks++;

    // time sorting
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (unsigned int phase = 1; phase <= k; phase++)
        for (unsigned int step = phase; step >= 1; step--)
            bitonicSort<<<numBlocks, NUM_THREADS>>>(gpuVecPtr, size, phase, step);
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
