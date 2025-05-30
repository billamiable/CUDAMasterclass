#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ int myAtomicAdd(int *address, int incr) 
{   
	int expected = *address;    
	// check stored value should be equal to expected value (second parameter)
	// if true, fill address with new value (third parameter)
	// else, no change made to address (another thread changes value between code)
	// in either case, return value stored at address
	int oldValue = atomicCAS(address, expected, expected + incr);
													
	while (oldValue != expected) 
	{       
		// in this case, updated expected value with new value
		expected = oldValue;   
		// same as before, essentially taking all operations into consideration
		oldValue = atomicCAS(address, expected, expected + incr);
	}    
	return oldValue;
}

__global__ void new_atomic_add_test(int *ptr)
{
	myAtomicAdd(ptr,1);
}

int main()
{
	int value = 23;
	int SIZE = sizeof(int);
	int ref = -1;

	int *d_val;
	cudaMalloc((void**)&d_val, SIZE);
	cudaMemcpy(d_val, &value, SIZE, cudaMemcpyHostToDevice);
	new_atomic_add_test << <1, 32 >> > (d_val);
	cudaDeviceSynchronize();
	cudaMemcpy(&ref, d_val, SIZE, cudaMemcpyDeviceToHost);

	printf("Updated value : %d \n", ref);

	cudaDeviceReset();
	return 0;
}