#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <sstream>
#include <iostream>
#include "common.cuh"
using namespace std;

void usage()
{
  cout << "usage: quicksort [k]" << endl;
  cout << "where 2^k is the size of the vector to generate for sorting" << endl;
  exit(1);
}

__global__ void partition(int *arr, int *low, int *high, int n, int *out_low, int *out_high, int *new_size)
{
  int pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos < n)
  {
    int hi = high[pos];
    int lo = low[pos];
    int pivot = arr[hi];
    int i = (lo - 1);
    int temp;
    for (int j = lo; j <= hi - 1; j++)
    {
      if (arr[j] <= pivot)
      {
        i++;
        temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
      }
    }
    temp = arr[i + 1];
    arr[i + 1] = arr[hi];
    arr[hi] = temp;
    int p = (i + 1);

    if (p - 1 > lo)
    {
      int ind = atomicAdd(new_size, 1);
      out_low[ind] = lo;
      out_high[ind] = p - 1;
    }
    if (p + 1 < hi)
    {
      int ind = atomicAdd(new_size, 1);
      out_low[ind] = p + 1;
      out_high[ind] = hi;
    }
  }
}

void quicksort(int arr[], int l, int h)
{
  int *low_indices = (int *)malloc((h - l + 1) * sizeof(int));
  int *high_indices = (int *)malloc((h - l + 1) * sizeof(int));

  int top = -1, *device_data, *dev_low_indices, *dev_high_indices;

  low_indices[++top] = l;
  high_indices[top] = h;

  cudaMalloc(&device_data, (h - l + 1) * sizeof(int));
  cudaMemcpy(device_data, arr, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_low_indices, (h - l + 1) * sizeof(int));
  cudaMemcpy(dev_low_indices, low_indices, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_high_indices, (h - l + 1) * sizeof(int));
  cudaMemcpy(dev_high_indices, high_indices, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);

  int num_threads = 1;
  int num_blocks = 1;
  int num_subarrays = 1;

  int *device_lows, *device_highs, *device_size;
  cudaMalloc(&device_lows, (h - l + 1) * sizeof(int));
  cudaMalloc(&device_highs, (h - l + 1) * sizeof(int));
  cudaMalloc(&device_size, sizeof(int));

  while (num_subarrays > 0)
  {
    int new_size = 0;
    cudaMemcpy(device_size, &new_size, sizeof(int), cudaMemcpyHostToDevice);
    partition<<<num_blocks, num_threads>>>(device_data, dev_low_indices, dev_high_indices, num_subarrays, device_lows, device_highs, device_size);

    cudaMemcpy(&new_size, device_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(dev_low_indices, device_lows, new_size * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_high_indices, device_highs, new_size * sizeof(int), cudaMemcpyDeviceToDevice);

    if (new_size < NUM_THREADS)
    {
      num_threads = new_size;
    }
    else
    {
      num_threads = NUM_THREADS;
      num_blocks = new_size / num_threads + (new_size % num_threads == 0 ? 0 : 1);
    }
    num_subarrays = new_size;
    cudaMemcpy(arr, device_data, (h - l + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  }

  cudaFree(device_data);
  cudaFree(dev_low_indices);
  cudaFree(dev_high_indices);
  cudaFree(device_lows);
  cudaFree(device_highs);
  cudaFree(device_size);
  free(low_indices);
  free(high_indices);
}

int main(int argc, char **argv)
{
  istringstream ss(argv[1]);
  unsigned int k;
  if (!(ss >> k) || k > sizeof(size_t) * 8 - 1)
  {
    usage();
  }

  size_t n = 1 << k;
  size_t bytes = n * sizeof(int);
  int *in = (int *)malloc(bytes);

  for (size_t i = 0; i < n; i++)
  {
    in[i] = rand();
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  quicksort(in, 0, n - 1);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  checkCudaError();

  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout << milliseconds << endl;

#ifdef DEBUG
  if (!isSorted(in, n))
    cout << "vect is not sorted!" << endl;
#endif

  free(in);
  return EXIT_SUCCESS;
}
