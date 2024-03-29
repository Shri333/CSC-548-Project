/**
 * @file radixsort.cu
 * @author Henry Sneed & Bonan Huang, Jinlan Gao, Xiaoming Li
 * @brief Implementation based on Empirically Optimized Radix Sort for 
 * GPU by Bonan Huang, Jinlan Gao and Xiaoming Li with code adapted from https://github.com/qcuong98/GPURadixSort
 * @date 2023-04-01
 *
 */
#include <bits/stdc++.h>
#include <stdint.h>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include "common.cuh"

#define CTA_SIZE 4
#define BLOCKSIZE 256
#define K_BITS 8
#define N_STREAMS 16
#define N_BINS (1 << K_BITS)
#define ELEMENTS_PER_BLOCK (2 * CTA_SIZE * BLOCKSIZE)
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) + ((n) >> LOG_NUM_BANKS))
using namespace std;

void printArray(uint32_t *a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

__device__ __forceinline__ uint32_t getBin(uint32_t val, uint32_t bit, uint32_t nBins)
{
    return (val >> bit) & (nBins - 1);
}

__global__ void scanBlkKernel(uint32_t *src, int n, uint32_t *out, uint32_t *blkSums)
{
    extern __shared__ uint32_t s[];
    uint32_t *localScan = s;
    uint32_t *localScanCTA = localScan + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK);

    int ai = threadIdx.x;
    int bi = threadIdx.x + blockDim.x;

    uint32_t first = ELEMENTS_PER_BLOCK * blockIdx.x;
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
    {
        int pos = first + i;
        localScan[CONFLICT_FREE_OFFSET(i)] = pos < n ? src[pos] : 0;
    }
    __syncthreads();

    uint32_t tempA[CTA_SIZE], tempB[CTA_SIZE];
#pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i)
    {
        tempA[i] = localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)];
        tempB[i] = localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)];
        if (i)
        {
            tempA[i] += tempA[i - 1];
            tempB[i] += tempB[i - 1];
        }
    }

    // compute scan
    localScanCTA[CONFLICT_FREE_OFFSET(ai)] = tempA[CTA_SIZE - 1];
    localScanCTA[CONFLICT_FREE_OFFSET(bi)] = tempB[CTA_SIZE - 1];
    __syncthreads();

// reduction phase
#pragma unroll
    for (int stride = 1, d = BLOCKSIZE; stride <= BLOCKSIZE; stride <<= 1, d >>= 1)
    {
        if (threadIdx.x < d)
        {
            int cur = 2 * stride * (threadIdx.x + 1) - 1;
            int prev = cur - stride;
            localScanCTA[CONFLICT_FREE_OFFSET(cur)] += localScanCTA[CONFLICT_FREE_OFFSET(prev)];
        }
        __syncthreads();
    }
// post-reduction phase
#pragma unroll
    for (int stride = BLOCKSIZE >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1)
    {
        if (threadIdx.x < d - 1)
        {
            int prev = 2 * stride * (threadIdx.x + 1) - 1;
            int cur = prev + stride;
            localScanCTA[CONFLICT_FREE_OFFSET(cur)] += localScanCTA[CONFLICT_FREE_OFFSET(prev)];
        }
        __syncthreads();
    }

    uint32_t lastScanA = ai ? localScanCTA[CONFLICT_FREE_OFFSET(ai - 1)] : 0;
    uint32_t lastScanB = localScanCTA[CONFLICT_FREE_OFFSET(bi - 1)];
    __syncthreads();

#pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i)
    {
        tempA[i] += lastScanA;
        tempB[i] += lastScanB;

        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)] = tempA[i];
        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)] = tempB[i];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
    {
        if (first + i < n)
            out[first + i] = localScan[CONFLICT_FREE_OFFSET(i)];
    }

    if (threadIdx.x == blockDim.x - 1)
    {
        blkSums[blockIdx.x] = tempB[CTA_SIZE - 1];
    }
}

__global__ void sumPrefixBlkKernel(uint32_t *out, int n, uint32_t *blkSums)
{
    uint32_t lastBlockSum = blockIdx.x > 0 ? blkSums[blockIdx.x - 1] : 0;
    uint32_t first = ELEMENTS_PER_BLOCK * blockIdx.x;
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
    {
        if (first + i < n)
            out[first + i] += lastBlockSum;
    }
}

__global__ void reduceKernel(uint32_t *in, int n, uint32_t *out)
{
    int id_in = blockDim.x * blockIdx.x + threadIdx.x;
    if (id_in < n)
        out[id_in] -= in[id_in];
}

void computeScanArray(uint32_t *d_in, uint32_t *d_out, int n, dim3 elementsPerBlock, dim3 blockSize)
{
    dim3 gridSize((n - 1) / elementsPerBlock.x + 1);

    uint32_t *d_blkSums;
    cudaMalloc(&d_blkSums, gridSize.x * sizeof(uint32_t));
    uint32_t *d_sum_blkSums;
    cudaMalloc(&d_sum_blkSums, gridSize.x * sizeof(uint32_t));

    scanBlkKernel<<<gridSize, blockSize, CONFLICT_FREE_OFFSET((2 * CTA_SIZE + 2) * blockSize.x) * sizeof(uint32_t)>>>(d_in, n, d_out, d_blkSums);
    if (gridSize.x != 1)
    {
        computeScanArray(d_blkSums, d_sum_blkSums, gridSize.x, elementsPerBlock, blockSize);
    }
    sumPrefixBlkKernel<<<gridSize, blockSize>>>(d_out, n, d_sum_blkSums);

    cudaFree(d_sum_blkSums);
    cudaFree(d_blkSums);
}

__global__ void scatterKernel(uint32_t *src, int n, uint32_t *dst, uint32_t *histScan, int bit, uint32_t *count)
{
    extern __shared__ uint32_t start[];
    uint32_t first = N_BINS * blockIdx.x;
    for (int i = threadIdx.x; i < N_BINS; i += blockDim.x)
    {
        start[CONFLICT_FREE_OFFSET(i)] = histScan[first + i];
    }
    __syncthreads();

    first = ELEMENTS_PER_BLOCK * blockIdx.x;
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
    {
        if (first + i < n)
        {
            uint32_t val = src[first + i];
            uint32_t st = start[CONFLICT_FREE_OFFSET(getBin(val, bit, N_BINS))];
            uint32_t equalsBefore = count[first + i];
            uint32_t pos = st + equalsBefore - 1;
            dst[pos] = val;
        }
    }
}

__global__ void sortLocalKernel(uint32_t *src, int n, int bit, uint32_t *count, uint32_t *hist, int start_pos = 0)
{
    extern __shared__ uint32_t s[];
    uint32_t *localSrc = s;
    uint32_t *localBin = localSrc + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK);
    uint32_t *localScan = localBin + CONFLICT_FREE_OFFSET(2 * BLOCKSIZE);
    uint32_t *s_hist = localScan + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK);

    int ai = threadIdx.x;
    int bi = threadIdx.x + blockDim.x;
    uint32_t first = ELEMENTS_PER_BLOCK * blockIdx.x;

    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
    {
        int pos = first + i;
        localSrc[CONFLICT_FREE_OFFSET(i)] = pos < n ? src[pos] : UINT_MAX;
    }
    __syncthreads();

    // radix sort with k = 1
    uint32_t tempA[CTA_SIZE], tempB[CTA_SIZE];
#pragma unroll
    for (int b = 0; b < K_BITS; ++b)
    {
        int blockBit = bit + b;
        uint32_t valA = 0, valB = 0;
#pragma unroll
        for (int i = 0; i < CTA_SIZE; ++i)
        {
            uint32_t thisA = getBin(tempA[i] = localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)], bit, N_BINS);
            valA += (tempA[i] >> blockBit & 1);
            uint32_t thisB = getBin(tempB[i] = localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)], bit, N_BINS);
            valB += (tempB[i] >> blockBit & 1);
        }

        // compute scan
        localScan[CONFLICT_FREE_OFFSET(ai)] = valA;
        localScan[CONFLICT_FREE_OFFSET(bi)] = valB;
        __syncthreads();

// reduction phase
#pragma unroll
        for (int stride = 1, d = BLOCKSIZE; stride <= BLOCKSIZE; stride <<= 1, d >>= 1)
        {
            if (threadIdx.x < d)
            {
                int cur = 2 * stride * (threadIdx.x + 1) - 1;
                int prev = cur - stride;
                localScan[CONFLICT_FREE_OFFSET(cur)] += localScan[CONFLICT_FREE_OFFSET(prev)];
            }
            __syncthreads();
        }
// post-reduction phase
#pragma unroll
        for (int stride = BLOCKSIZE >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1)
        {
            if (threadIdx.x < d - 1)
            {
                int prev = 2 * stride * (threadIdx.x + 1) - 1;
                int cur = prev + stride;
                localScan[CONFLICT_FREE_OFFSET(cur)] += localScan[CONFLICT_FREE_OFFSET(prev)];
            }
            __syncthreads();
        }

        // scatter
        int n0 = ELEMENTS_PER_BLOCK - localScan[CONFLICT_FREE_OFFSET(2 * blockDim.x - 1)];

        valA = localScan[CONFLICT_FREE_OFFSET(ai)];
        valB = localScan[CONFLICT_FREE_OFFSET(bi)];
#pragma unroll
        for (int i = CTA_SIZE - 1; i >= 0; --i)
        {
            if (tempA[i] >> blockBit & 1)
                localSrc[CONFLICT_FREE_OFFSET(n0 + valA - 1)] = tempA[i];
            else
                localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i - valA)] = tempA[i];
            valA -= (tempA[i] >> blockBit & 1);

            if (tempB[i] >> blockBit & 1)
                localSrc[CONFLICT_FREE_OFFSET(n0 + valB - 1)] = tempB[i];
            else
                localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i - valB)] = tempB[i];
            valB -= (tempB[i] >> blockBit & 1);
        }

        __syncthreads();
    }

    uint32_t countA[CTA_SIZE], countB[CTA_SIZE];
#pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i)
    {
        tempA[i] = getBin(localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)], bit, N_BINS);
        tempB[i] = getBin(localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)], bit, N_BINS);
        countA[i] = countB[i] = 1;
        if (i)
        {
            if (tempA[i] == tempA[i - 1])
                countA[i] += countA[i - 1];
            if (tempB[i] == tempB[i - 1])
                countB[i] += countB[i - 1];
        }
    }

    localScan[CONFLICT_FREE_OFFSET(ai)] = countA[CTA_SIZE - 1];
    localScan[CONFLICT_FREE_OFFSET(bi)] = countB[CTA_SIZE - 1];
    localBin[CONFLICT_FREE_OFFSET(ai)] = tempA[CTA_SIZE - 1];
    localBin[CONFLICT_FREE_OFFSET(bi)] = tempB[CTA_SIZE - 1];
    __syncthreads();

// reduction phase
#pragma unroll
    for (int stride = 1, d = BLOCKSIZE; stride <= BLOCKSIZE; stride <<= 1, d >>= 1)
    {
        if (threadIdx.x < d)
        {
            int cur = 2 * stride * (threadIdx.x + 1) - 1;
            int prev = cur - stride;
            cur = CONFLICT_FREE_OFFSET(cur);
            prev = CONFLICT_FREE_OFFSET(prev);
            if (localBin[cur] == localBin[prev])
                localScan[cur] += localScan[prev];
        }
        __syncthreads();
    }
// post-reduction phase
#pragma unroll
    for (int stride = BLOCKSIZE >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1)
    {
        if (threadIdx.x < d - 1)
        {
            int prev = 2 * stride * (threadIdx.x + 1) - 1;
            int cur = prev + stride;
            cur = CONFLICT_FREE_OFFSET(cur);
            prev = CONFLICT_FREE_OFFSET(prev);
            if (localBin[cur] == localBin[prev])
                localScan[cur] += localScan[prev];
        }
        __syncthreads();
    }

    uint32_t lastBinA = localBin[CONFLICT_FREE_OFFSET(ai - 1)];
    uint32_t lastBinB = localBin[CONFLICT_FREE_OFFSET(bi - 1)];
    uint32_t lastScanA = ai ? localScan[CONFLICT_FREE_OFFSET(ai - 1)] : 0;
    uint32_t lastScanB = localScan[CONFLICT_FREE_OFFSET(bi - 1)];
    __syncthreads();

#pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i)
    {
        if (tempA[i] == lastBinA)
            countA[i] += lastScanA;

        if (tempB[i] == lastBinB)
            countB[i] += lastScanB;

        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)] = countA[i];
        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)] = countB[i];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
    {
        int pos = first + i;
        if (pos < n)
        {
            count[pos] = localScan[CONFLICT_FREE_OFFSET(i)];
            src[pos] = localSrc[CONFLICT_FREE_OFFSET(i)];
        }
    }

    for (int idx = threadIdx.x; idx < N_BINS; idx += blockDim.x)
        s_hist[CONFLICT_FREE_OFFSET(idx)] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x)
    {
        int pos = first + i;
        if (pos < n)
        {
            uint32_t thisBin = getBin(localSrc[CONFLICT_FREE_OFFSET(i)], bit, N_BINS);
            if (pos == n - 1 || i == ELEMENTS_PER_BLOCK - 1 || thisBin != getBin(localSrc[CONFLICT_FREE_OFFSET(i + 1)], bit, N_BINS))
                s_hist[CONFLICT_FREE_OFFSET(thisBin)] = localScan[CONFLICT_FREE_OFFSET(i)];
        }
    }
    __syncthreads();

    first = (blockIdx.x + start_pos) * N_BINS;
    for (int digit = threadIdx.x; digit < N_BINS; digit += blockDim.x)
        hist[first + digit] = s_hist[CONFLICT_FREE_OFFSET(digit)];
}

__global__ void transpose(uint32_t *iMatrix, uint32_t *oMatrix, int rows, int cols)
{
    __shared__ int s_blkData[32][33];
    int iR = blockIdx.x * blockDim.x + threadIdx.y;
    int iC = blockIdx.y * blockDim.y + threadIdx.x;
    s_blkData[threadIdx.y][threadIdx.x] = (iR < rows && iC < cols) ? iMatrix[iR * cols + iC] : 0;
    __syncthreads();
    // Each block write data efficiently from SMEM to GMEM
    int oR = blockIdx.y * blockDim.y + threadIdx.y;
    int oC = blockIdx.x * blockDim.x + threadIdx.x;
    if (oR < cols && oC < rows)
        oMatrix[oR * rows + oC] = s_blkData[threadIdx.x][threadIdx.y];
}

void radixsort(const uint32_t *in, int n, uint32_t *out)
{
    uint32_t *d_src;
    uint32_t *d_dst;
    uint32_t *d_hist;
    uint32_t *d_histScan;
    uint32_t *d_count;
    cudaMalloc(&d_src, n * sizeof(uint32_t));
    cudaMalloc(&d_count, n * sizeof(uint32_t));
    cudaMalloc(&d_dst, n * sizeof(uint32_t));

    // Compute block and grid size for scan and scatter phase
    dim3 blockSize(BLOCKSIZE);
    dim3 elementsPerBlock(ELEMENTS_PER_BLOCK);
    dim3 gridSize((n - 1) / elementsPerBlock.x + 1);
    dim3 blockSizeTranspose(32, 32);
    dim3 gridSizeTransposeHist((gridSize.x - 1) / blockSizeTranspose.x + 1, (N_BINS - 1) / blockSizeTranspose.x + 1);
    dim3 gridSizeTransposeHistScan((N_BINS - 1) / blockSizeTranspose.x + 1, (gridSize.x - 1) / blockSizeTranspose.x + 1);

    int histSize = N_BINS * gridSize.x;
    cudaMalloc(&d_hist, 2 * histSize * sizeof(uint32_t));
    cudaMalloc(&d_histScan, 2 * histSize * sizeof(uint32_t));
    dim3 gridSizeScan((histSize - 1) / blockSize.x + 1);

    cudaStream_t *streams = (cudaStream_t *)malloc(N_STREAMS * sizeof(cudaStream_t));
    for (int i = 0; i < N_STREAMS; ++i)
    {
        cudaStreamCreate(&streams[i]);
        checkCudaError();
    }
    int len = (gridSize.x - 1) / N_STREAMS + 1;
    for (int i = 0; i < N_STREAMS; ++i)
    {
        int cur_pos = i * len * elementsPerBlock.x;
        if (cur_pos >= n)
            break;
        int cur_len = min(len * elementsPerBlock.x, n - i * len * elementsPerBlock.x);
        dim3 cur_gridSize((cur_len - 1) / elementsPerBlock.x + 1);
        cudaMemcpyAsync(d_src + cur_pos, in + cur_pos, cur_len * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, streams[i]);
        checkCudaError();
        sortLocalKernel<<<cur_gridSize, blockSize, CONFLICT_FREE_OFFSET((4 * CTA_SIZE + 2) * blockSize.x + N_BINS) * sizeof(uint32_t), streams[i]>>>(d_src + cur_pos, cur_len, 0, d_count + cur_pos, d_hist + histSize, i * len);
    }

    for (int bit = 0; bit < 32; bit += K_BITS)
    {
        if (bit)
        {
            sortLocalKernel<<<gridSize, blockSize, CONFLICT_FREE_OFFSET((4 * CTA_SIZE + 2) * BLOCKSIZE + N_BINS) * sizeof(uint32_t)>>>(d_src, n, bit, d_count, d_hist + histSize);
        }

        transpose<<<gridSizeTransposeHist, blockSizeTranspose>>>(d_hist + histSize, d_hist, gridSize.x, N_BINS);

        // compute hist scan
        computeScanArray(d_hist, d_histScan + histSize, histSize, elementsPerBlock, blockSize);
        reduceKernel<<<gridSizeScan, blockSize>>>(d_hist, histSize, d_histScan + histSize);
        checkCudaError();
        transpose<<<gridSizeTransposeHistScan, blockSizeTranspose>>>(d_histScan + histSize, d_histScan, N_BINS, gridSize.x);
        checkCudaError();

        // scatter
        scatterKernel<<<gridSize, blockSize, CONFLICT_FREE_OFFSET(N_BINS) * sizeof(uint32_t)>>>(d_src, n, d_dst, d_histScan, bit, d_count);
        uint32_t *tmp = d_src;
        d_src = d_dst;
        d_dst = tmp;
    }

    cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_hist);
    cudaFree(d_histScan);
}

void usage()
{
    cout << "usage: quicksort [k]" << endl;
    cout << "where 2^k is the size of the vector to generate for sorting" << endl;
    exit(1);
}

int main(int argc, char **argv)
{
    // read k from argv[1] where 2^k is the size of the vector to generate
    istringstream ss(argv[1]);
    unsigned int k;
    if (!(ss >> k) || k > sizeof(size_t) * 8 - 1)
    {
        usage();
    }

    size_t n = 1 << k;
    size_t bytes = n * sizeof(uint32_t);
    uint32_t *in = (uint32_t *)malloc(bytes);
    uint32_t *out = (uint32_t *)malloc(bytes);
    for (int i = 0; i < n; i++)
    {
        in[i] = rand();
    }
    cout << "Sorting vector of size " << n << "..." << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    radixsort(in, n, out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkCudaError();

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << " ms" << endl;

#ifdef DEBUG
    if (!isSorted(out, n))
        cout << "vect is not sorted!" << endl;
#endif
    free(in);
    free(out);
    return EXIT_SUCCESS;
}
