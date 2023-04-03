#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <thrust/scan.h>
#include <sstream>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "common.cuh"
#define THREADS_PER_BLOCK 256
using namespace std;

void usage()
{
  cout << "usage: quicksort [k]" << endl;
  cout << "where 2^k is the size of the vector to generate for sorting" << endl;
  exit(1);
}

/**
 * @brief Device helper function to swap elements in an array
 *
 * @param arr array of floating point values
 * @param index_a the position of the element to swap with the value at index_b
 * @param index_b the position of the element to swap with the value at index_a
 */
__device__ void swap(float *arr, int index_a, int index_b)
{
  float temp = arr[index_a];
  arr[index_a] = arr[index_b];
  arr[index_b] = temp;
}

/**
 * @brief Reorders elements less than pivot to the left of the pivot
 * and elements greater than the pivot to the right for the given partition.
 * The input array values are randomly generated so always choosing
 * the rightmost element is equivalent to taking a random partition
 * @param arr the array to reorder
 * @param left the left most index of the subarray to partition
 * @param right the right most index of the subarray to partition
 * @return __device__
 */
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

/**
 * @brief Recursive quicksort implementation
 *
 * @param arr the array to sort
 * @param left the left most index of the array to sort
 * @param right the right most index of the array to sort
 */
__device__ void device_quicksort(float *arr, int left, int right)
{
  if (left < right)
  {
    int pivot_index = device_partition(arr, left, right);
    device_quicksort(arr, left, pivot_index - 1);
    device_quicksort(arr, pivot_index + 1, right);
  }
}

/**
 * @brief Responsible for calling device quicksort function
 *
 * @param arr the array to sort
 * @param size the number of elements in the array
 */
__global__ void quicksort_kernel(float *arr, int size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx == 0)
  {
    device_quicksort(arr, 0, size - 1);
  }
}

/**
 * @brief Performs quicksort on the given array
 *
 * @param arr the array to sort
 * @param size the number of elements in the array
 */
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

/**
 * @brief Generates an array of floating point values and performs
 * quicksort on the array.
 *
 * @param argc Two arguments are expects: ./quicksort and #floats to generate and sort
 * @param argv The array of arguments
 * @return int The exit code
 */
int main(int argc, char **argv)
{
  // read k from argv[1] where 2^k is the size of the vector to generate
  istringstream ss(argv[1]);
  unsigned int k;
  if (!(ss >> k) || k > sizeof(size_t) * 8 - 1)
  {
    usage();
  }

  // generate vector
  size_t size = 1 << k;
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
