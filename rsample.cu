#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/adjacent_difference.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <curand_kernel.h>
#include "common.cuh"
#define MAX_BLOCK_SIZE 1024
#define THREADS_PER_BLOCK 256
#define SEED 1234
using namespace std;

__global__ void initCurandState(curandState *state)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(SEED, idx, 0, &state[idx]);
}

__global__ void generate_samples(float *samples, const float *data, int data_size, int num_samples, curandState *state)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_samples)
    return;

  int randIndex = curand(&state[idx]) % data_size;
  samples[idx] = data[randIndex];
}

__global__ void partition_data(float *data, const float *pivots, int *bucket_counts, int data_size, int num_pivots)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= data_size)
    return;

  float value = data[idx];
  int bucket_idx = 0;

  while (bucket_idx < num_pivots && value >= pivots[bucket_idx])
  {
    bucket_idx++;
  }

  int offset = atomicAdd(&bucket_counts[bucket_idx], 1);
  data[data_size + bucket_idx * data_size + offset] = value;
}

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: samplesort <size>" << std::endl;
    exit(EXIT_FAILURE);
  }

  int size = std::stoi(argv[1]);
  thrust::host_vector<float> h_vec = genVec(size);

  // The number of samples is pretty arbitrary so this can be adjusted later
  int num_samples = static_cast<int>(sqrt(size));
  thrust::device_vector<float> d_vec(size * (num_samples + 1));
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());
  printf("\nUnsorted:\n");
  for (size_t i = 0; i < h_vec.size(); ++i)
  {
    printf("\t%f", h_vec[i]);
  }
  printf("\n");
  cout << "Sorting vector of size " << size << "..." << endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  thrust::device_vector<curandState> d_curand_states(num_samples);
  initCurandState<<<(num_samples + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(d_curand_states.data()));
  cudaDeviceSynchronize();
  checkCudaError();

  // device samples
  thrust::device_vector<float> d_samples(num_samples);
  generate_samples<<<(num_samples + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(d_samples.data()), thrust::raw_pointer_cast(d_vec.data()), size, num_samples, thrust::raw_pointer_cast(d_curand_states.data()));
  cudaDeviceSynchronize();
  checkCudaError();

  // sort the array of samples
  thrust::sort(d_samples.begin(), d_samples.end());

  thrust::device_vector<float> d_pivots(num_samples - 1);
  thrust::adjacent_difference(d_samples.begin() + 1, d_samples.end(), d_pivots.begin());
  thrust::device_vector<int> d_bucket_counts(num_samples, 0);

  partition_data<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(d_vec.data()), thrust::raw_pointer_cast(d_pivots.data()), thrust::raw_pointer_cast(d_bucket_counts.data()), size, num_samples - 1);
  cudaDeviceSynchronize();
  checkCudaError();

  float *data_ptr = thrust::raw_pointer_cast(d_vec.data()) + size;
  for (int i = 0; i < num_samples; ++i)
  {
    int bucket_size = d_bucket_counts[i];
    thrust::sort(thrust::device_pointer_cast(data_ptr + i * size), thrust::device_pointer_cast(data_ptr + i * size + bucket_size));
  }

  thrust::device_vector<float> d_sorted(size);
  thrust::device_vector<float> d_temp(size);

  int sorted_data_end = 0;
  for (int i = 0; i < num_samples; ++i)
  {
    int bucket_size = d_bucket_counts[i];
    thrust::sort(thrust::device_pointer_cast(data_ptr + i * size), thrust::device_pointer_cast(data_ptr + i * size + bucket_size));

    int new_sorted_data_end = sorted_data_end + bucket_size;
    thrust::merge(thrust::device_pointer_cast(data_ptr + i * size), thrust::device_pointer_cast(data_ptr + i * size + bucket_size), d_sorted.begin(), d_sorted.begin() + sorted_data_end, d_temp.begin());
    thrust::copy(d_temp.begin(), d_temp.begin() + new_sorted_data_end, d_sorted.begin());
    sorted_data_end = new_sorted_data_end;
  }

  thrust::copy(d_sorted.begin(), d_sorted.end(), h_vec.begin());

  printf("\nSorted: \n");
  for (size_t i = 0; i < h_vec.size(); ++i)
  {
    printf("\t%f", h_vec[i]);
  }
  printf("\n");

  // get time to sort
  cudaEventSynchronize(stop);
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // print out time to sort
  cout << "Time: " << milliseconds << " ms" << endl;

#ifdef DEBUG
  if (!sorted(h_vec))
    cout << "vec is not sorted!" << endl;
#endif
  return 0;
}