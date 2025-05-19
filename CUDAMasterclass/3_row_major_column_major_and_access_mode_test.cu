#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

// for block with 1024 elements, only 1 thread block is enough
#define BDIMX 32
#define BDIMY 32

__global__ void setRowReadCol(int * out)
{
	// now I understand why opencv's image matrix tends to use image[iy][ix]
	// it's designed for better memory access pattern
	__shared__ int tile[BDIMY][BDIMX];

	// no longer need blockId here because we only use 1 thread block
	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	//store to the shared memory
	// set is row major
	tile[threadIdx.y][threadIdx.x] = idx;

	//waiting for all the threads in thread block to reach this point
	__syncthreads();

	//load from shared memory
	// note that here x and y positions are switched
	// read is column major
	out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadRow(int * out)
{
	__shared__ int tile[BDIMY][BDIMX];

	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	//store to the shared memory
	// difference is here
	// set is column major
	// will have bank conflict due to bank is partitioned using column
	tile[threadIdx.x][threadIdx.y] = idx;

	//waiting for all the threads in thread block to reach this point
	__syncthreads();

	//load from shared memory
	// read is row major
	out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadRow(int * out)
{
	__shared__ int tile[BDIMY][BDIMX];

	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	//store to the shared memory
	// always y at first
	tile[threadIdx.y][threadIdx.x] = idx;

	//waiting for all the threads in thread block to reach this point
	__syncthreads();

	//load from shared memory
	// always y at first
	out[idx] = tile[threadIdx.y][threadIdx.x];
}

int main(int argc, char **argv)
{
	int memconfig = 0;
	if (argc > 1)
	{
		memconfig = atoi(argv[1]);
	}


	// dynamically configurable using cuda device set shared memory config
	if (memconfig == 1)
	{
		// 64 bit
		printf("setting 64 bit access mode\n");
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	}
	else
	{
		// 32 bit
		printf("setting 32 bit access mode\n");
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	}

	
	cudaSharedMemConfig pConfig;
	cudaDeviceGetSharedMemConfig(&pConfig);
	printf("%d\n", pConfig);
	// seems no change for GeoForce 2080 Ti
	printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");
	

	// set up array size 2048
	int nx = BDIMX;
	int ny = BDIMY;

	bool iprintf = 0;
	
	if (argc > 2) iprintf = atoi(argv[1]);

	size_t nBytes = nx * ny * sizeof(int);

	// execution configuration
	dim3 block(BDIMX, BDIMY);
	dim3 grid(1, 1);
	printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
		block.y);

	// allocate device memory
	int *d_C;
	cudaMalloc((int**)&d_C, nBytes);
	int *gpuRef = (int *)malloc(nBytes);

	cudaMemset(d_C, 0, nBytes);
	setColReadRow << <grid, block >> >(d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	if (iprintf)  printData("set col read col   ", gpuRef, nx * ny);

	cudaMemset(d_C, 0, nBytes);
	setRowReadRow << <grid, block >> >(d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	if (iprintf)  printData("set row read row   ", gpuRef, nx * ny);

	cudaMemset(d_C, 0, nBytes);
	setRowReadCol << <grid, block >> >(d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	if (iprintf)  printData("set row read col   ", gpuRef, nx * ny);

	// free host and device memory
	cudaFree(d_C);
	free(gpuRef);

	// reset device
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
