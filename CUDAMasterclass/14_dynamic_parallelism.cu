#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void dynamic_parallelism_check(int size, int depth)
{
    // this is similar to dfs
	printf(" Depth : %d - tid : %d \n", depth, threadIdx.x);

	if (size == 1)
		return;

	if (threadIdx.x == 0)
	{
		// launch kernel inside kernel - dymaic parallelism
		dynamic_parallelism_check << <1, size / 2 >> > (size / 2, depth + 1);
	}
}

int main(int argc, char** argv)
{
	// grid, thread block - input parameter
	dynamic_parallelism_check << <1, 16 >> > (16,0);
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}