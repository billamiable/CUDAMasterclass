#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include "cuda_common.cuh"

__global__ void k1()
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid == 0)
	{
		printf("This is a test 1 \n");
	}
}

int main(int argc, char ** argv)
{
	int size = 1 << 15;

	cudaStream_t stm1,stm2,stm3;
	cudaStreamCreate(&stm1);
	cudaStreamCreate(&stm2);
	cudaStreamCreate(&stm3);

	cudaEvent_t event1;
	// don't need to record timing, thus to improve performance
	cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);

	dim3 block(128);
	dim3 grid(size / block.x);
	
	k1 << <grid, block, 0, stm1 >> > ();
	// create event after launching first kernel
	cudaEventRecord(event1, stm1);
	// use wait operation to ask stream3 to wait event to finish
	// therefore, it's expected that stream3 kernel launch after stream1 kernel finishes
	cudaStreamWaitEvent(stm3, event1, 0);

	k1 << <grid, block, 0, stm2 >> > ();
	k1 << <grid, block, 0, stm3 >> > ();

	// destroy event
	cudaEventDestroy(event1);

	cudaStreamDestroy(stm1);
	cudaStreamDestroy(stm2);
	cudaStreamDestroy(stm3);

	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
