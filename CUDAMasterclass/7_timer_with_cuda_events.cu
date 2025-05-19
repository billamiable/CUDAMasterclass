#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include "cuda_common.cuh"

__global__ void event_test()
{
	double sum = 0.0;
	for (int i = 0; i < 1000; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

int main(int argc, char ** argv)
{
	int size = 1 << 12;

	dim3 block(128);
	dim3 grid(size / block.x);

	cudaEvent_t start, end;

	// create events
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// push event to queue before kernel launch
	cudaEventRecord(start);

	event_test << < grid,block >>> ();

	// push event to queue after kernel launch
	cudaEventRecord(end);
	// sync to make sure event is finished
	// mainly to block host operation until event is reached and finished
	cudaEventSynchronize(end);

	float time;
	// check time cost
	cudaEventElapsedTime(&time, start, end);

	printf("Kernel execution time using events : %f \n",time);

	// destroy events
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	cudaDeviceReset();
	return 0;
}