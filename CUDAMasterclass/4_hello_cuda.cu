#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_cuda()
{
	printf("Hello CUDA world \n");
}

int main()
{
	int nx, ny;
	nx = 16; // number of threads in x dim
	ny = 4;  // number of threads in y dim

	// for missing number, default is 1
	dim3 block(8, 2);
	// 2, 2
	dim3 grid(nx / block.x,ny / block.y);

	// print out 64 times
	// launched asynced
	hello_cuda << < grid, block >> > ();
	// need sync to wait the result before proceeding
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}