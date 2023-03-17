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

// sample size for sample sort (should be a power of 2 and an even divisor of NUM_THREADS)
#define SAMPLE_SIZE 64

// log base 2 of sample size
#define LOG_SAMPLE_SIZE 6

void usage() {
    cout << "usage: dsample [k]" << endl;
    cout << "where 2^k is the size of the vector to generate for sorting" << endl;
    exit(1);
}

// kernel for sorting with bitonic sort
__global__ void bitonicSort(float vec[], size_t size, unsigned int phase, unsigned int step) {
    size_t idx = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    bitonicSwap(vec, size, phase, step, idx);
}

// sort a partition of the global vector on each block using bitonic sort
__global__ void localBitonicSort(float vec[], size_t size) {
    size_t idx = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;

    // copy partition of vector into shared memory
    __shared__ float sharedVec[NUM_THREADS];
    sharedVec[threadIdx.x] = vec[idx];
    __syncthreads();

    // bitonic sort this sub-vector/partition
    for (unsigned int phase = 1; phase <= LOG_NUM_THREADS; phase++) {
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

// sample equidistant values from vec into samples
__global__ void sample(float vec[], float samples[]) {
    size_t idx = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;
    if (idx % SAMPLE_SIZE == 0) {
        samples[idx / SAMPLE_SIZE] = vec[idx];
    }
}

// searches for the index in vec where the given num should be located
__device__ size_t binarySearch(float vec[], size_t size, float num) {
    size_t left = 0, right = size - 1;
    while (left < right) {
        size_t mid = left + (right - left) / 2; // to avoid overflow
        if (vec[mid] < num) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

// calculates bucket sizes by indexing each global sample in each subvector
__global__ void calcBucketSizes(float vec[], float globalSamples[], size_t bucketSizes[]) {
    size_t idx = ((size_t) blockDim.x) * blockIdx.x + threadIdx.x;

    // copy subvec into shared memory
    __shared__ float sharedVec[NUM_THREADS];
    __shared__ size_t sampleIndices[SAMPLE_SIZE];
    sharedVec[threadIdx.x] = vec[idx];
    __syncthreads();

    // binary search indices of each global sample in subvec
    if (threadIdx.x < SAMPLE_SIZE) {
        size_t sampleIdx = binarySearch(sharedVec, NUM_THREADS, globalSamples[threadIdx.x]);
        sampleIndices[threadIdx.x] = sampleIdx;
    }
    __syncthreads();

    // calculate size of each bucket based on sampleIndices
    // if (threadIdx.x < SAMPLE_SIZE) {
    //     if (threadIdx.x == SAMPLE_SIZE - 1) {
    //         bucketSizes[gridDim.x * threadIdx.x + blockIdx.x] 
    //             = NUM_THREADS - sampleIndices[SAMPLE_SIZE - 1];
    //     } else {
    //         bucketSizes[gridDim.x * threadIdx.x + blockIdx.x] 
    //             = sampleIndices[threadIdx.x] - sampleIndices[threadIdx.x - 1];
    //     }
    // }
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
        // sort the vector with bitonic sort instead when the vector is small enough
        for (unsigned int phase = 1; phase <= k; phase++) {
            for (unsigned int step = phase; step >= 1; step--) {
                bitonicSort<<<1, size / 2>>>(gpuVecPtr, size, phase, step);
            }
        }
    } else {
        // sort a partition of the vector (subvec) on each block w/ bitonic sort
        size_t numBlocks = size / NUM_THREADS;
        localBitonicSort<<<numBlocks, NUM_THREADS>>>(gpuVecPtr, size);

        // sample locally
        size_t localSamplesSize = numBlocks * SAMPLE_SIZE;
        thrust::device_vector<float> localSamples(localSamplesSize);
        float* localSamplesPtr = thrust::raw_pointer_cast(localSamples.data());
        sample<<<numBlocks, NUM_THREADS>>>(gpuVecPtr, localSamplesPtr);

        // sort local samples with bitonic sort
        numBlocks = max((size_t) 1, (localSamplesSize / 2) / NUM_THREADS);
        size_t numThreads = min(localSamplesSize / 2, (size_t) NUM_THREADS);
        for (unsigned int phase = 1; phase <= k - LOG_NUM_THREADS + LOG_SAMPLE_SIZE; phase++) {
            for (unsigned int step = phase; step >= 1; step--) {
                bitonicSort<<<numBlocks, numThreads>>>(localSamplesPtr, localSamplesSize, phase, step);
            }
        }

        // sample globally
        thrust::device_vector<float> globalSamples(SAMPLE_SIZE);
        float* globalSamplesPtr = thrust::raw_pointer_cast(globalSamples.data());
        numBlocks = max((size_t) 1, localSamplesSize / NUM_THREADS);
        numThreads = min(localSamplesSize, (size_t) NUM_THREADS);
        sample<<<numBlocks, numThreads>>>(localSamplesPtr, globalSamplesPtr);

        // calculate sizes of buckets (sample indexing)
        thrust::device_vector<size_t> bucketSizes(localSamplesSize);
        size_t* bucketSizesPtr = thrust::raw_pointer_cast(bucketSizes.data());
        numBlocks = size / NUM_THREADS;
        calcBucketSizes<<<numBlocks, NUM_THREADS>>>(gpuVecPtr, globalSamplesPtr, bucketSizesPtr);
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
