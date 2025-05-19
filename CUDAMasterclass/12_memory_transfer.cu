#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__ void mem_trs_test(int * input)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("tid : %d , gid : %d, value : %d \n",threadIdx.x,gid,input[gid]);
}

// won't access out of bound
__global__ void mem_trs_test2(int * input, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if(gid < size)
		printf("tid : %d , gid : %d, value : %d \n", threadIdx.x, gid, input[gid]);
}

// TODO extend to 3d 3d
__global__ void mem_trs_test3(int * input, int size)
{
	int tid = blockDim.x * threadIdx.y + threadIdx.x;

	int num_threads_in_a_block = blockDim.x * blockDim.y;
	int block_offset = blockIdx.x * num_threads_in_a_block;

	int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
	int row_offset = num_threads_in_a_row * blockIdx.y;

	int gid = tid + block_offset + row_offset;

	// printf("blockIdx.x : %d, blockIdx.y : %d, threadIdx.x : %d, gid : %d - data : %d \n",
	// 	blockIdx.x, blockIdx.y, tid, gid, data[gid]);

	if(gid < size)
		printf("tid : %d , gid : %d, value : %d \n", threadIdx.x, gid, input[gid]);
}

int main()
{
	int size = 16;
	int byte_size = size * sizeof(int);

	int * h_input;
	// key to understand malloc
	h_input = (int*)malloc(byte_size);

	time_t t;
	srand((unsigned)time(&t));
	// prepare random int values
	for (int i = 0; i < size; i++)
	{
		h_input[i] = (int)(rand() & 0xff);
	}

	int * d_input;
	// pointer to pointer
	cudaMalloc((void**)&d_input, byte_size);

	cudaMemcpy(d_input,h_input,byte_size,cudaMemcpyHostToDevice);

	// dim3 block(32);
	// dim3 grid(5);
	dim3 block(2,2);
	dim3 grid(2,2);

	// mem_trs_test2 << <grid, block >> > (d_input,size);
	mem_trs_test3 << <grid, block >> > (d_input,size);
	cudaDeviceSynchronize();

	cudaFree(d_input);
	free(h_input);

	cudaDeviceReset();
	return 0;
}