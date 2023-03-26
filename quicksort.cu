#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "common.cuh"
#define THREADS_PER_BLOCK 256
using namespace std;

__device__ void swap(float *arr, int index_a, int index_b)
{
  float temp = arr[index_a];
  arr[index_a] = arr[index_b];
  arr[index_b] = temp;
}

// we know the input array is random so always choosing
// the rightmost element is equivalent to taking a random partition
__device__ int device_partition(float *arr, int left, int right)
{
  float pivot = arr[right];
  int i = left - 1;
  for (int k = left; k <= right; ++k)
  {
    if (arr[k] < pivot)
    {
      swap(arr, ++i, k);
    }
  }
  swap(arr, i + 1, right);
  return (i + 1);
}

__device__ void device_quicksort(float *arr, int left, int right)
{
  if (left < right)
  {
    int pivot_index = device_partition(arr, left, right);
    device_quicksort(arr, left, pivot_index - 1);
    device_quicksort(arr, pivot_index + 1, right);
  }
}

__global__ void quicksort_kernel(float *arr, int size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx == 0)
  {
    device_quicksort(arr, 0, size - 1);
  }
}

void quicksort(float *arr, int size)
{
  float *device_data;
  cudaMalloc(&device_data, size * sizeof(float));
  cudaMemcpy(device_data, arr, size * sizeof(float), cudaMemcpyHostToDevice);

  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  quicksort_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(device_data, size);
  cudaDeviceSynchronize();

  cudaMemcpy(arr, device_data, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(device_data);
}

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: samplesort <size>" << std::endl;
    exit(EXIT_FAILURE);
  }

  int size = std::stoi(argv[1]);
  thrust::host_vector<float> host_vec = genVec(size);
  thrust::device_vector<float> device_vec(size);
  thrust::copy(host_vec.begin(), host_vec.end(), device_vec.begin());

  cout << "Sorting vector of size " << size << "..." << endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  quicksort(thrust::raw_pointer_cast(device_vec.data()), size);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  checkCudaError();

  thrust::copy(device_vec.begin(), device_vec.end(), host_vec.begin());

  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cout << "Time: " << milliseconds << " ms" << endl;

#ifdef DEBUG
  if (!sorted(host_vec))
    cout << "vect is not sorted!" << endl;
#endif
  return 0;
}
