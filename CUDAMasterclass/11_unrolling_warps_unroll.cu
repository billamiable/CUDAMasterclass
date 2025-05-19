#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// it's essentially based on interleaved pair approach
__global__ void reduction_kernel_warp_unrolling(int * int_array,
	int * temp_array, int size)
{
	int tid = threadIdx.x;
	
	//element index for this thread
	int index = blockDim.x * blockIdx.x  + threadIdx.x;

	//local data pointer
	int * i_data = int_array + blockDim.x * blockIdx.x ;

	// this is important to stop at offset = 64
	// because if it's already 32, then unrolling will cause warp divergence
	// blockDim.x means number of threads in a thread block
	for (int offset = blockDim.x/2; offset >= 64; offset = offset/2)
	{
		if (tid < offset)
		{
			i_data[tid] += i_data[tid + offset];
		}
		__syncthreads();
	}

	// this won't cause warp divergence
	// 当出现warp divergence时，并行计算的效率会受到影响，因为GPU需要多次执行相同的warp来确保所有线程都完成任务。
	if (tid < 32)
	{
		// volatile identifier makes sure that it won't use cache
		// 使用volatile可以明确告诉编译器，这些变量的值可能会在不被预期的情况下改变，从而阻止编译器进行这类优化
		// vsmem可能是一个缩写，代表“volatile shared memory”
		// 虽然直接在i_data上进行操作通常也是可行的，但通过使用volatile修饰的新变量vsmem，程序员可以进一步强调这些内存访问的特殊性，从而减少编译器进行不期望优化的可能性
		volatile int * vsmem = i_data;
		// below is just simple unrolling
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	// finally, save result and prepare to do a thread-block-level sum outside GPU
	if (tid == 0)
	{
		temp_array[blockIdx.x] = i_data[0];
	}
}

int main(int argc, char ** argv)
{
	printf("Running parallel reduction with warp unrolling kernel \n");

	// make sure same as before
	int size = 1 << 27;
	int byte_size = size * sizeof(int);
	int block_size = 128; // 4 warps

	int * h_input, *h_ref;
	h_input = (int*)malloc(byte_size);

	initialize(h_input, size, INIT_RANDOM);

	int cpu_result = reduction_cpu(h_input, size);

	dim3 block(block_size); // one thread block has 128 threads
	dim3 grid(size / block_size); // one grid has 32768 thread blocks

	printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);

	int temp_array_byte_size = sizeof(int)* grid.x;

	h_ref = (int*)malloc(temp_array_byte_size);

	int * d_input, *d_temp;
	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));

	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
		cudaMemcpyHostToDevice));

	reduction_kernel_warp_unrolling <<< grid, block >> > (d_input, d_temp, size);

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));

	int gpu_result = 0;
	// grid.x means how many thread blocks in a grid
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