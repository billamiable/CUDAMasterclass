#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include "cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//reduction neighbored pairs kernel
__global__ void reduction_neighbored_pairs(int * input, 
	int * temp, int size)
{
	int tid = threadIdx.x;
	// get global id, relative simple because only have 1d grid
	// within each thread block, thread id is unique because only 1d thread block
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	// no need to calculate larger than size
	if (gid > size)
		return;

	for (int offset = 1; offset <= blockDim.x/2; offset *= 2)
	{
		// won't have issue because this part will constrain the thread that participate in calculation
		// TODO but this will result in warp divergence!!
		if (tid % (2 * offset) == 0)
		{
			// won't this exceed the max of array? - answer above
			// in-place sum, directly change input array
			input[gid] += input[gid + offset];
		}
		// important to make sure each iteration's result is finished before the next iteration starts
		__syncthreads();
	}

	// get partial result, for each thread block in grid, get temporary result
	// because for each thread block, final result is stored in the first element of array
	// therefore, directly get its tid==0's result
	if (tid == 0)
	{
		temp[blockIdx.x] = input[gid];
	}
}

int main(int argc, char ** argv)
{
	printf("Running neighbored pairs reduction kernel \n");

	int size = 1 << 27; //128 Mb of data
	int byte_size = size * sizeof(int);
	int block_size = 128;

	// host side input array, and array to copy result back
	int * h_input, *h_ref;
	// int* is type cast, which means converts the pointer to int* of allocated memory
	h_input = (int*)malloc(byte_size);

	initialize(h_input, size, INIT_RANDOM);

	//get the reduction result from cpu
	clock_t cpu_start, cpu_end, gpu_start, gpu_end;

	cpu_start = clock();
	int cpu_result = reduction_cpu(h_input,size);
	cpu_end = clock();

	printf("CPU execution time : %4.6f \n",
		(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));

	dim3 block(block_size);
	dim3 grid(size/ block.x);

	// grid.x : 1048576, block.x : 128
	// such a big number for thread block # in the grid
	printf("Kernel launch parameters | grid.x : %d, block.x : %d \n",
		grid.x, block.x);

	// used to get computation result from device, need further calculation on host side
	int temp_array_byte_size = sizeof(int) * grid.x;
	h_ref = (int*)malloc(temp_array_byte_size);

	// identifier for device side input array, and temporary calculation result
	int * d_input, *d_temp;

	// cudaMalloc expects a void** as its first parameter
	// cudaMalloc requires a void** because it needs to update the caller's pointer with the address of the allocated memory
	// cudaMalloc needs to access and modify the pointer variable itself, not just a copy of its value
	// To pass d_input so that cudaMalloc can modify it, you need to provide the address of d_input -> so use &
	gpuErrchk(cudaMalloc((void**)&d_input,byte_size));
	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));

	// set a specified memory area on the GPU to a particular value
	gpuErrchk(cudaMemset(d_temp, 0 , temp_array_byte_size));

	// copy input from host to device
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size, 
		cudaMemcpyHostToDevice));

	reduction_neighbored_pairs << <grid, block >> > (d_input,d_temp, size);

	gpuErrchk(cudaDeviceSynchronize());

	// copy result from device to host
	cudaMemcpy(h_ref,d_temp, temp_array_byte_size,
		cudaMemcpyDeviceToHost);

	// finish the final sum after partial results are obtained
	int gpu_result = 0;

	// only need to sum number of thread blocks
	for (int i = 0; i < grid.x; i++)
	{
		gpu_result += h_ref[i];
	}

	//validity check
	compare_results(gpu_result, cpu_result);

	gpuErrchk(cudaFree(d_temp));
	gpuErrchk(cudaFree(d_input));

	free(h_ref);
	free(h_input);

	gpuErrchk(cudaDeviceReset());
	return 0;
}