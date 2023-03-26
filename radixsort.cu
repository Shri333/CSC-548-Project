#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/partition.h>
#include <thrust/count.h>
#include <thrust/reverse.h>
#include <thrust/extrema.h>

#include "common.cuh"

using namespace std;

__global__ void float_to_int(float *arr, int *int_arr, int size)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index >= size)
    return;

  int_arr[index] = __float_as_int(arr[index]);
}

__global__ void int_to_float(int *int_arr, float *arr, int size)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index >= size)
    return;

  arr[index] = __int_as_float(int_arr[index]);
}

// Radix Sort kernel
__global__ void radixsort_kernel(int *arr, int *output, int *count, int size, int exp)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index >= size)
    return;

  int bin = (arr[index] / exp) % 10;
  atomicAdd(&count[bin], 1);
  __syncthreads();

  // Prefix sum
  if (threadIdx.x == 0)
  {
    int sum = 0;
    for (int i = 0; i < 10; i++)
    {
      int tmp = count[i];
      count[i] = sum;
      sum += tmp;
    }
  }
  __syncthreads();

  int pos = atomicAdd(&count[bin], 1);
  output[pos] = arr[index];
  __syncthreads();

  arr[index] = output[index];
}

__host__ void radixsort(float *arr, int size)
{
  thrust::device_vector<float> device_arr(size);

  cudaMemcpy(device_arr.data().get(), arr, size * sizeof(float), cudaMemcpyHostToDevice);

  thrust::stable_sort(device_arr.begin(), device_arr.end(), [] __device__(float a, float b)
                      {
    int a_int = __float_as_int(a);
    int b_int = __float_as_int(b);
    if ((a_int ^ b_int) < 0) {
      return a_int < b_int;
    } else {
      if (a_int < 0) {
        return (a_int & 0x7FFFFFFF) > (b_int & 0x7FFFFFFF);
      } else {
        return (a_int & 0x7FFFFFFF) < (b_int & 0x7FFFFFFF);
      }
    } });

  cudaMemcpy(arr, device_arr.data().get(), size * sizeof(float), cudaMemcpyDeviceToHost);
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

  radixsort(thrust::raw_pointer_cast(device_vec.data()), size);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  thrust::copy(device_vec.begin(), device_vec.end(), host_vec.begin());

  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cout << "Time: " << milliseconds << " ms" << endl;

#ifdef DEBUG
  if (!sorted(host_vec))
    cout << "vec is not sorted!" << endl;
#endif
  return 0;
}
