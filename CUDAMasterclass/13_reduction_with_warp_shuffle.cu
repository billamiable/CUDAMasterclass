#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 128 //1024
#define FULL_MASK 0xffffffff

// basically the best algo so far
// this is the same as reduction_smem in 7_reduction_smem.cu
template<unsigned int iblock_size>
__global__ void reduction_smem_benchmark(int * input, int * temp, int size)
{
	// using shared memory
	__shared__ int smem[BLOCK_SIZE];
	int tid = threadIdx.x;
	int * i_data = input + blockDim.x * blockIdx.x;

	smem[tid] = i_data[tid];

	__syncthreads();

	// using template
	// in-place reduction in shared memory   
	if (iblock_size >= 1024 && tid < 512)
		smem[tid] += smem[tid + 512];
	__syncthreads();

	if (iblock_size >= 512 && tid < 256)
		smem[tid] += smem[tid + 256];
	__syncthreads();

	if (iblock_size >= 256 && tid < 128)
		smem[tid] += smem[tid + 128];
	__syncthreads();

	if (iblock_size >= 128 && tid < 64)
		smem[tid] += smem[tid + 64];
	__syncthreads();

	// using warp unrolling
	//unrolling warp
	if (tid < 32)
	{
		volatile int * vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0)
	{
		temp[blockIdx.x] = smem[0];
	}
}

template<unsigned int iblock_size>
__global__ void reduction_smem_warp_shfl(int * input, int * temp, int size)
{
	__shared__ int smem[BLOCK_SIZE];
	int tid = threadIdx.x;
	int * i_data = input + blockDim.x * blockIdx.x;

	smem[tid] = i_data[tid];

	__syncthreads();

	// in-place reduction in shared memory
	if (iblock_size >= 1024 && tid < 512)
		smem[tid] += smem[tid + 512];
	__syncthreads();

	if (iblock_size >= 512 && tid < 256)
		smem[tid] += smem[tid + 256];
	__syncthreads();

	if (iblock_size >= 256 && tid < 128)
		smem[tid] += smem[tid + 128];
	__syncthreads();

	if (iblock_size >= 128 && tid < 64)
		smem[tid] += smem[tid + 64];
	__syncthreads();

	// changed part1: this part is also newly added
	// because warp shuffling can only operate within a single warp
	// cannot write in below section because below section only operate within a warp
	// here it is actually larger than 1 warp
	if (iblock_size >= 64 && tid < 32)
		smem[tid] += smem[tid + 32];
	__syncthreads();


	// changed part2: the following part
	// replace warp unrolling with warp shuffling + unrolling
	// essentially, it's a change on how to calculate array sum within a warp
	// use local variable to store temporary summed value
	int local_sum = smem[tid];

	//unrolling warp
	if (tid < 32)
	{
		// since we are getting value for only first element in the array
		// yes, here we have many redundant computation, especially those at the tail, but it's fine
		// 0xffffffff - mask is no use
		// local sum always sum by itself
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 16);
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 8);
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 4);
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 2);
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 1);
	}

	if (tid == 0)
	{
		// use local summed value instead
		temp[blockIdx.x] = local_sum;
	}
}


int main(int argc, char ** argv)
{
	printf("Running parallel reduction with complete unrolling kernel \n");

	int size = 1 << 27; //22
	int byte_size = size * sizeof(int);
	int block_size = BLOCK_SIZE;

	int * h_input, *h_ref;
	h_input = (int*)malloc(byte_size);

	initialize(h_input, size);

	clock_t cpu_start, cpu_end, gpu_start, gpu_end;

	cpu_start = clock();
	int cpu_result = reduction_cpu(h_input, size);
	cpu_end = clock();

	printf("CPU execution time : %4.6f \n",
		(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));

	dim3 block(block_size);
	dim3 grid((size / block_size));

	printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);

	int temp_array_byte_size = sizeof(int)* grid.x;

	h_ref = (int*)malloc(temp_array_byte_size);

	int * d_input, *d_temp;

	printf(" \nreduction with shared memory\n ");
	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));

	printf("GPU \n");
	gpu_start = clock();

	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,cudaMemcpyHostToDevice));
	
	switch (block_size)
	{
	case 1024:
		reduction_smem_benchmark <1024> << < grid, block >> > (d_input, d_temp, size);
		break;
	case 512:
		reduction_smem_benchmark <512> << < grid, block >> > (d_input, d_temp, size);
		break;
	case 256:
		reduction_smem_benchmark <256> << < grid, block >> > (d_input, d_temp, size);
		break;
	case 128:
		reduction_smem_benchmark <128> << < grid, block >> > (d_input, d_temp, size);
		break;
	case 64:
		reduction_smem_benchmark <64> << < grid, block >> > (d_input, d_temp, size);
		break;
	}

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));

	int gpu_result = 0;
	for (int i = 0; i < grid.x; i++)
	{
		gpu_result += h_ref[i];
	}

	gpu_end = clock();
	// compare_results(gpu_result, cpu_result);
	print_time_using_host_clock(gpu_start, gpu_end);

	//warp shuffle implementation
	printf(" \nreduction with warp shuffle instructions \n ");

	printf("GPU \n");
	gpu_start = clock();
	
	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));

	switch (block_size)
	{
	case 1024:
		reduction_smem_warp_shfl <1024> << < grid, block >> > (d_input, d_temp, size);
		break;
	case 512:
		reduction_smem_warp_shfl <512> << < grid, block >> > (d_input, d_temp, size);
		break;
	case 256:
		reduction_smem_warp_shfl <256> << < grid, block >> > (d_input, d_temp, size);
		break;
	case 128:
		reduction_smem_warp_shfl <128> << < grid, block >> > (d_input, d_temp, size);
		break;
	case 64:
		reduction_smem_warp_shfl <64> << < grid, block >> > (d_input, d_temp, size);
		break;
	}
	// reduction_smem_warp_shfl <128> << < grid, block >> > (d_input, d_temp, size);

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));

	gpu_result = 0;
	for (int i = 0; i < grid.x; i++)
	{
		gpu_result += h_ref[i];
	}

	gpu_end = clock();
	// compare_results(gpu_result, cpu_result);
	print_time_using_host_clock(gpu_start, gpu_end);

	gpuErrchk(cudaFree(d_input));
	gpuErrchk(cudaFree(d_temp));
	free(h_input);
	free(h_ref);

	gpuErrchk(cudaDeviceReset());
	return 0;
}