#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_threadIds()
{
	printf("threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d \n",
		threadIdx.x,threadIdx.y,threadIdx.z);
}

int main()
{
	int nx, ny;
	nx = 16;
	ny = 16;

	dim3 block(8,8); // each block have 8 threads in x and y dim
	dim3 grid(nx/ block.x, ny/block.y); // 2*2 grid

	print_threadIds << <grid,block >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
