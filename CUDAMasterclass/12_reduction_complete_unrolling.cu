#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 通过完全展开归约过程，可以显著减少循环迭代的次数。每次循环迭代都需要进行条件判断和迭代变量更新
// 这些操作会消耗一定的计算资源。完全展开可以消除这些开销，从而提高程序的执行效率。
// 但实际上效果可能变差
__global__ void reduction_kernel_complete_unrolling(int * int_array,
	int * temp_array, int size)
{
	// this implementation doesn't involve thread block unrolling!
	int tid = threadIdx.x;

	//element index for this thread
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	//local data pointer
	int * i_data = int_array + blockDim.x * blockIdx.x;

	// it means that it could handle at most 1024 threads per thread block
	// actually that's the upper bound of thread num for x/y dim for a thread block
	// even more, x*y*z thread num cannot exceed 1024, which means this code can work for any setup
	// using only >= instead of ==  can easily solve this issue 
	if (blockDim.x == 1024 && tid < 512)
		// different tid will operate on different data
		i_data[tid] += i_data[tid + 512];
	__syncthreads();

	// this original part is actually not correct!
	if (blockDim.x >= 512 && tid < 256)
		// different tid will operate on different data
		i_data[tid] += i_data[tid + 256];
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
		// different tid will operate on different data
		i_data[tid] += i_data[tid + 128];
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
		i_data[tid] += i_data[tid + 64];
	__syncthreads();

	if (tid < 32)
	{
		volatile int * vsmem = i_data;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0)
	{
		temp_array[blockIdx.x] = i_data[0];
	}
}

int main(int argc, char ** argv)
{
	printf("Running parallel reduction with complete unrolling kernel \n");

	int size = 1 << 27;
	int byte_size = size * sizeof(int);
	// int block_size = 1024;
	int block_size = 128;

	int * h_input, *h_ref;
	h_input = (int*)malloc(byte_size);

	initialize(h_input, size, INIT_RANDOM);

	int cpu_result = reduction_cpu(h_input, size);

	dim3 block(block_size);
	dim3 grid(size / block_size);

	printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);

	int temp_array_byte_size = sizeof(int)* grid.x;

	h_ref = (int*)malloc(temp_array_byte_size);

	int * d_input, *d_temp;
	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));

	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
		cudaMemcpyHostToDevice));

	reduction_kernel_complete_unrolling <<< grid, block >> > (d_input, d_temp, size);

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));

	int gpu_result = 0;
	for (int i = 0; i < grid.x; i++)
	{
		gpu_result += h_ref[i];
	}

	compare_results(gpu_result, cpu_result);

	gpuErrchk(cudaFree(d_input));
	gpuErrchk(cudaFree(d_temp));
	free(h_input);
	free(h_ref);

	gpuErrchk(cudaDeviceReset());
	return 0;
}