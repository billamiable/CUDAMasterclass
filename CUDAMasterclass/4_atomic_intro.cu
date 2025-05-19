#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void incr(int *ptr)
{
	// due to no use of threadid here, so all threads are accessing the same memory address using broadcast
	// therefore below code has output as 1
	// if having multiple thread blocks, the output will be undefined (tested still 1)
	// int temp = *ptr;
	// temp = temp + 1;
	// *ptr = temp;

	// to enable processing between competing threads, use atomic operation
	// below code has output as 32
	atomicAdd(ptr,1);
}

int main()
{
	int value = 0;	
	int SIZE = sizeof(int);
	int ref = -1;

	int *d_val;
	cudaMalloc((void**)&d_val, SIZE);
	cudaMemcpy(d_val, &value, SIZE, cudaMemcpyHostToDevice);
	incr << <1, 32 >> > (d_val);
	cudaDeviceSynchronize();
	cudaMemcpy(&ref,d_val,SIZE, cudaMemcpyDeviceToHost);

	printf("Updated value : %d \n",ref);

	cudaDeviceReset();
	return 0;
}