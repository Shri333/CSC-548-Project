// implementation of Batcher's odd-even mergesort
// based on "Sorting Networks and their Applications" by Batcher
// author: Shrihan Dadi (sdadi2)
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "common.cuh"
using namespace std;

void usage() {
    cout << "usage: batcher [k]" << endl;
    cout << "where 2^k is the size of the vector to generate for sorting" << endl;
    exit(1);
}

// kernel for batcher's sorting network swaps
__global__ void batcherOddEvenSwap(float vec[], size_t size, size_t phase, size_t step) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    // index calculation algorithm based on https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
    for (size_t j = step % phase; j < size - step; j += 2 * step) {
        for (size_t i = 0; i < step; i++) {
            if (idx == i + j && (i + j) / (2 * phase) == (i + j + step) / (2 * phase)) {
                cmpSwap(vec, i + j, i + j + step);
                return;
            }
        }
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
    size_t numBlocks = size / NUM_THREADS;
    if (size % NUM_THREADS != 0) {
        numBlocks++;
    }

    // time sorting
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (size_t phase = 1; phase < size; phase <<= 2) {
        for (size_t step = phase; step >= 1; step >>= 2) {
            batcherOddEvenSwap<<<numBlocks, NUM_THREADS>>>(gpuVecPtr, size, phase, step);
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
