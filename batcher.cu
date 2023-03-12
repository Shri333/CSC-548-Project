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
__global__ void batcherOddEvenSwap(float vec[], size_t size, unsigned int phase, unsigned int step) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    // partner calculation algorithm based on https://gist.github.com/Bekbolatov/c8e42f5fcaa36db38402
    size_t partner = idx ^ (1 << (phase - 1));
    if (step > 1) {
        size_t scale = 1 << (phase - step);
        size_t box = 1 << step;
        size_t scaledIdx = idx / scale - (idx / scale / box) * box;
        if (scaledIdx == 0 || scaledIdx == box - 1) {
            return;
        }
        partner = (scaledIdx & 1) == 0 ? idx - scale : idx + scale;
    }

    cmpSwap(vec, idx, partner);
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
    for (unsigned int phase = 1; phase <= k; phase++) {
        for (unsigned int step = phase; step >= 1; step--) {
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
