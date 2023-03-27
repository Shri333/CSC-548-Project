#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>
#include "common.cuh"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#define THREADS_PER_BLOCK 256
#define NUM_BITS_IN_INT 32
#define RADIX 10
using namespace std;

/**
 * @brief Convert floats to ints with bitwise operation to ensure negative
 * floats are represented correctly in the array.
 *
 * @param arr The input float array
 * @param int_arr The output float array
 * @param size the number of elements in the input array
 */
__global__ void float_to_int(float *arr, int *int_arr, int size)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index >= size)
    return;

  int_arr[index] = __float_as_int(arr[index]) ^ ((-(arr[index] < 0)) | 0x80000000);
}

/**
 * @brief Convert integer array to float array
 *
 * @param int_arr The sorted integer array
 * @param arr The sorted float array
 * @param size The number of elements in the sorted integer array
 */
__global__ void int_to_float(int *int_arr, float *arr, int size)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index >= size)
    return;

  arr[index] = __int_as_float(int_arr[index] ^ ((int_arr[index] >> 31) - 1 | 0x80000000));
}

/**
 * @brief Performs counting sort as a subroutine for radix sort
 * on groups of bits
 *
 * @param arr a pointer to the array of integers to sort
 * @param output the pointer to the output array where the sorted
 *               integers are stored
 * @param count keeps track of the frequency of each digit
 * @param size the size of the input array used to determine
 *             the range of indices to operate on
 * @param exp the position of the current group of bits
 * @return __global__
 */
__global__ void countingsort(int *arr, int *output, int *count, int size, int exp)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index >= size)
    return;

  // right shift the int at the current index by exp bits to extract
  // the lsb, and only extract 9 lsbs
  // subtract 1 so that the leading two's complement 1 is removed from negative numbers
  int bin = ((arr[index] >> exp) & 0x1FF) - 1;
  atomicAdd(&count[bin], 1);
  __syncthreads();

  // Thread 0 performs prefix sum
  if (threadIdx.x == 0)
  {
    int sum = 0;
    for (int i = 0; i < RADIX; i++)
    {
      int tmp = count[i];
      count[i] = sum;
      sum += tmp;
    }
  }
  __syncthreads();

  // place input elements in their positions in the output array using
  // the count array
  int pos = atomicAdd(&count[bin], 1);
  output[pos] = arr[index];
  __syncthreads();

  arr[index] = output[index];
}

/**
 * @brief Performs radix sort on array of floating point values
 *
 * @param arr the array of floats
 * @param size the number of elements in the array
 */
__host__ void radixsort(float *arr, int size)
{
  float *device_arr;
  int *device_int_arr;
  int *device_output;
  int *device_count;

  cudaMalloc(&device_arr, size * sizeof(float));
  cudaMalloc(&device_int_arr, size * sizeof(int));
  cudaMalloc(&device_output, size * sizeof(int));
  cudaMalloc(&device_count, RADIX * sizeof(int));

  cudaMemcpy(device_arr, arr, size * sizeof(float), cudaMemcpyHostToDevice);

  float_to_int<<<(size + 255) / 256, 256>>>(device_arr, device_int_arr, size);
  checkCudaError();

  int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  thrust::device_ptr<int> device_int_ptr(device_int_arr);
  int max = thrust::reduce(device_int_ptr, device_int_ptr + size, std::numeric_limits<int>::min(), thrust::maximum<int>());

  for (int exp = 1; exp < NUM_BITS_IN_INT; exp += 9)
  {
    cudaMemset(device_count, 0, RADIX * sizeof(int));
    countingsort<<<num_blocks, THREADS_PER_BLOCK>>>(device_int_arr, device_output, device_count, size, exp);
    cudaDeviceSynchronize();
  }

  int_to_float<<<num_blocks, THREADS_PER_BLOCK>>>(device_int_arr, device_arr, size);
  checkCudaError();

  cudaMemcpy(arr, device_arr, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(device_arr);
  cudaFree(device_int_arr);
  cudaFree(device_output);
  cudaFree(device_count);
}

/**
 * @brief Performs radix sort on input arrays
 *
 * @param argc number of program arguments
 * @param argv array of program arguments
 * @return int exit status code
 */
int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: samplesort <size>" << std::endl;
    exit(EXIT_FAILURE);
  }

  int size = std::stoi(argv[1]);
  thrust::host_vector<float> host_vec = genVec(size);
  cout << "Sorting vector of size " << size << "..." << endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  radixsort(host_vec.data(), size);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cout << "Time: " << milliseconds << " ms" << endl;

#ifdef DEBUG
  if (!sorted(host_vec))
    cout << "vec is not sorted!" << endl;
#endif
  return 0;
}