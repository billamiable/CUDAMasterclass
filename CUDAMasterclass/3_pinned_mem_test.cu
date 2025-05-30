#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) 
{   								
	// memory size   128 MBs
	int isize = 1<<25;   
	int nbytes = isize * sizeof(float);
											
	// allocate the host memory   
    // option1: use pageable memory
	//float *h_a = (float *)malloc(nbytes);
    // option2: use pinned memory
	float *h_a;
	cudaMallocHost((float **)&h_a, nbytes);

	// allocate the device memory   
	float *d_a; 
   cudaMalloc((float **)&d_a, nbytes);
									
	// initialize the host memory   
	for(int i=0;i<isize;i++) 
		h_a[i] = 7;
									
	// transfer data from the host to the device   
	cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
									
	// transfer data from the device to the host   
	cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);
							
	// free memory   
	cudaFree(d_a);
    // option1: use free to free pageable memory
	//free(h_a);
    // option2: use cudaFreeHost to free pinned memory
	cudaFreeHost(h_a);
									
	// reset device    
	cudaDeviceReset();   
	return EXIT_SUCCESS;
}