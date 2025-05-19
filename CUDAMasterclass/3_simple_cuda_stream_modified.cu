#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

__global__ void stream_test_modified(int* in, int * out, int size)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid < size)
	{
		//THIS FOR LOOP IS ONLY FOR VISUALIZING PURPOSE  
		for (int i = 0; i < 25; i++)
		{
			out[gid] = in[gid] + (in[gid] - 1) * (gid % 10);
		}
	}
}

int main(int argc, char ** argv)
{
	int size = 1 << 18;
	int byte_size = size * sizeof(int);

	//initialize host pointer
	// add a second copy to see execution order for different streams
	int* h_in, *h_ref, *h_in2, *h_ref2;
	
	// first allocate pinned memory
	// seems to forget free pinned meory
	cudaMallocHost((void**)&h_in,byte_size);
	cudaMallocHost((void**)&h_ref, byte_size);
	cudaMallocHost((void**)&h_in2, byte_size);
	cudaMallocHost((void**)&h_ref2, byte_size);

	initialize(h_in, INIT_RANDOM);
	initialize(h_in2, INIT_RANDOM);


	//allocate device pointers
	int * d_in, *d_out, *d_in2, *d_out2;
	cudaMalloc((void**)&d_in, byte_size);
	cudaMalloc((void**)&d_out, byte_size);
	cudaMalloc((void**)&d_in2, byte_size);
	cudaMalloc((void**)&d_out2, byte_size);

	// create cuda streams
	cudaStream_t str,str2;
	cudaStreamCreate(&str);
	cudaStreamCreate(&str2);


	//kernel launch
	dim3 block(128);
	dim3 grid(size / block.x);
	
	//transfer data from host to device
	// add to add stream as parameter
	cudaMemcpyAsync(d_in, h_in, byte_size, cudaMemcpyHostToDevice,str);
	// third parameter is for dynamic shared memory, no use, so set to 0
	stream_test_modified << <grid, block,0,str >> > (d_in, d_out, size);
	cudaMemcpyAsync(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost,str);
	
	cudaMemcpyAsync(d_in2, h_in2, byte_size, cudaMemcpyHostToDevice, str2);
	stream_test_modified << <grid, block, 0, str2 >> > (d_in2, d_out2, size);
	cudaMemcpyAsync(h_ref2, d_out2, byte_size, cudaMemcpyDeviceToHost, str2);
	
	// need to sync stream
	// essentially, use cudaDeviceSynchronize should also be fine
	cudaStreamSynchronize(str);
	// then destroy stream in the end
	cudaStreamDestroy(str);

	cudaStreamSynchronize(str2);
	cudaStreamDestroy(str2);

	cudaDeviceReset();
	return 0;
}