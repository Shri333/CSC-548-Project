// implementation of common code used by all parallel sorting implementations
// author: Shrihan Dadi (sdadi2)
#include <cstddef>
#include <random>
#include <thrust/host_vector.h>
#include "common.cuh"
#include <algorithm>
using namespace std;

// seed for randomly generating vectors of floats
#define SEED 13

// lowest number that can be generated
#define LOWEST -1

// highest number that can be generated
#define HIGHEST 1

// function for generating a random vector of floats
thrust::host_vector<float> genVec(size_t size)
{
    // create seeded generator
    mt19937 gen(SEED);
    uniform_real_distribution<float> dist(LOWEST, HIGHEST);

    // create and fill vec with random values
    cout << "Generating random vector of size " << size << "..." << endl;
    thrust::host_vector<float> vec(size);
    for (size_t i = 0; i < size; i++)
    {
        vec[i] = dist(gen);
    }

    return vec;
}

// compares the values of the elements at the two indices (i and j)
// and swaps if vec[i] > vec[j]
__device__ void cmpSwap(float vec[], size_t i, size_t j)
{
    if (vec[i] > vec[j])
    {
        float tmp = vec[i];
        vec[i] = vec[j];
        vec[j] = tmp;
    }
}

// function for performing a normalized bitonic swap
// given a step and phase and thread index
__device__ void bitonicSwap(float vec[], size_t size, unsigned int phase, unsigned int step, size_t idx)
{
    size_t groupSize = 1 << step;
    size_t threadsPerGroup = groupSize / 2;
    size_t groupIdx = idx / threadsPerGroup;
    size_t i = groupIdx * groupSize + (idx % threadsPerGroup);
    size_t j = i + threadsPerGroup;
    if (step == phase)
    {
        // first step: normalized swap
        j = (groupSize * (groupIdx + 1) - 1) - (idx % threadsPerGroup);
    }
    cmpSwap(vec, i, j);
}

void checkCudaError()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Checks all devices shared memories and return the minimum
size_t minSharedMemPerBlock()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        std::cout << "No CUDA devices found" << std::endl;
        return 0;
    }
    size_t current_min = SIZE_MAX;
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        if (current_min > deviceProp.sharedMemPerBlock)
        {
            current_min = deviceProp.sharedMemPerBlock;
        }
    }
    return current_min;
}

// function for checking if a vector of floats is sorted or not
// only used in DEBUG mode
#ifdef DEBUG
bool sorted(const thrust::host_vector<float> &vec)
{
    bool res = true;

    bool all_zeros = std::all_of(vec.begin(), vec.end(), [](float elem)
                                 { return elem == 0; });

    // The vector should not be considered sorted if it's been 0 filled
    if (all_zeros)
    {
        return false;
    }

    for (size_t i = 0; i < vec.size() - 1; i++)
    {
        if (vec[i] > vec[i + 1])
        {
            cout << "> Index: " << i << "; Value: " << vec[i] << endl;
            cout << "< Index: " << i + 1 << "; Value: " << vec[i + 1] << endl;
            res = false;
        }
    }
    return res;
}
#endif
