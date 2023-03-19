#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include "common.cuh"

#define THREADS_PER_BLOCK 256
using namespace std;

__device__ void merge(float *input, float *output, int left_offset, int left_size, int right_offset, int right_size)
{
  int i = 0, j = 0, k = left_offset;

  while (i < left_size && j < right_size)
  {
    if (input[left_offset + i] < input[right_offset + j])
    {
      output[k++] = input[left_offset + i++];
    }
    else
    {
      output[k++] = input[right_offset + j++];
    }
  }

  while (i < left_size)
  {
    output[k++] = input[left_offset + i++];
  }

  while (j < right_size)
  {
    output[k++] = input[right_offset + j++];
  }
}

__global__ void merge_kern(float *data, float *result, int n, int width)
{
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  int left_offset = global_id * 2 * width;
  if (left_offset >= n)
  {
    return;
  }

  int left_size = min(width, n - left_offset);
  int right_offset = left_offset + width;
  int right_size = min(width, n - right_offset);

  merge(data, result, left_offset, left_size, right_offset, right_size);
}

void mergesort(thrust::device_vector<float> &device_data)
{
  int n = device_data.size();
  thrust::device_vector<float> device_output(n);

  float *raw_data = thrust::raw_pointer_cast(device_data.data());
  float *raw_result = thrust::raw_pointer_cast(device_output.data());

  for (int width = 1; width < n; width *= 2)
  {
    int num_blocks = (n + 2 * width * THREADS_PER_BLOCK - 1) / (2 * width * THREADS_PER_BLOCK);
    merge_kern<<<num_blocks, THREADS_PER_BLOCK>>>(raw_data, raw_result, n, width);
    cudaDeviceSynchronize();
    checkCudaError();
    cudaMemcpy(raw_data, raw_result, n * sizeof(float), cudaMemcpyDeviceToDevice);
    checkCudaError();
  }
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

  printf("\nUnsorted:\n");
  for (size_t i = 0; i < host_vec.size(); ++i)
  {
    printf("\t%f", host_vec[i]);
  }
  printf("\n");
  cout << "Sorting vector of size " << size << "..." << endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  mergesort(device_vec);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  checkCudaError();

  thrust::copy(device_vec.begin(), device_vec.end(), host_vec.begin());

  printf("\nSorted: \n");
  for (size_t i = 0; i < host_vec.size(); ++i)
  {
    printf("\t%f", host_vec[i]);
  }
  printf("\n");

  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cout << "Time: " << milliseconds << " ms" << endl;

#ifdef DEBUG
  if (!sorted(host_vec))
    cout << "vec is not sorted!" << endl;
#endif
  return 0;
}
