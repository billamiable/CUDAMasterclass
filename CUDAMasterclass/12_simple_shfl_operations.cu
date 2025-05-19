#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

#define ARRAY_SIZE 32
// #define ARRAY_SIZE 64

// shuffle broadcast 32 
// 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 
// 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
__global__ void test_shfl_broadcast_32(int * in, int *out)
{
	// value stored in register because defined within kernel
	int x = in[threadIdx.x];
	// setting 32 as width means broadcast to all threads within a warp
	// setting 3 as source lane id means to broadcast this particular thread's register value
	// source lane id = threadIdx.x % 32 = 3, then threadIdx.x = 3
	int y = __shfl_sync(0xffffffff, x, 3, 32);
	// broadcast value within warp
	out[threadIdx.x] = y;
}

// make sense, just divide warp into two pieces
// shuffle broadcast 16 
// 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 
// 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19
__global__ void test_shfl_broadcast_16(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_sync(0xffffffff, x, 3, 16);
	out[threadIdx.x] = y;
}

// shuffle up 
// 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 
// 0,1,2,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28
__global__ void test_shfl_up(int * in, int *out)
{
	int x = in[threadIdx.x];
	// for 0~2, the value stays the same
	// for larger than 2, the value equals to register value shifted by 3 offset
	int y = __shfl_up_sync(0xffffffff, x, 3);
	out[threadIdx.x] = y;
}

// similarily, the resuls is expected
// shuffle down 
// 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 
// 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,29,30,31
__global__ void test_shfl_down(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_down_sync(0xffffffff, x, 3);
	out[threadIdx.x] = y;
}


// compared with shift down, no register value remains the same
// more like a complete shift operation
// it's just simple, each thread use the register value from threadIdx.x+offset
// shift around 
// 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 
// 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,0,1
__global__ void test_shfl_shift_around(int * in, int *out, int offset)
{
	int x = in[threadIdx.x];
	int y = __shfl_sync(0xffffffff, x, threadIdx.x + offset);
	out[threadIdx.x] = y;
}

// easy to understand as well
// shuffle xor butterfly 
// 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 
// 1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16,19,18,21,20,23,22,25,24,27,26,29,28,31,30
__global__ void test_shfl_xor_butterfly(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_xor_sync(0xffffffff, x, 1, 32);
	out[threadIdx.x] = y;
}


int main(int argc, char ** argv)
{
	int size = ARRAY_SIZE;
	int byte_size = size * sizeof(int);

	int * h_in = (int*)malloc(byte_size);
	int * h_ref = (int*)malloc(byte_size);

	for (int i = 0; i < size; i++)
	{
		h_in[i] = i;
	}

	int * d_in, *d_out;

	cudaMalloc((int **)&d_in, byte_size);
	cudaMalloc((int **)&d_out, byte_size);

	dim3 block(size);
	dim3 grid(1);

	//broadcast 32
	printf("shuffle broadcast 32 \n");
	cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
	test_shfl_broadcast_32 << < grid, block >> > (d_in, d_out);
	cudaDeviceSynchronize();

	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);

	print_array(h_in, size);
	print_array(h_ref, size);

	//broadcast 16
	printf("shuffle broadcast 16 \n");
	cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
	test_shfl_broadcast_16 << < grid, block >> > (d_in, d_out);
	cudaDeviceSynchronize();

	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);

	print_array(h_in, size);
	print_array(h_ref, size);
	printf("\n");

	//up
	printf("shuffle up \n");
	cudaMemset(d_out, 0, byte_size);
	test_shfl_up << < grid, block >> > (d_in, d_out);
	cudaDeviceSynchronize();

	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);

	print_array(h_in, size);
	print_array(h_ref, size);
	printf("\n");

	//down
	printf("shuffle down \n");
	cudaMemset(d_out, 0, byte_size);
	test_shfl_down << < grid, block >> > (d_in, d_out);
	cudaDeviceSynchronize();

	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);

	print_array(h_in, size);
	print_array(h_ref, size);
	printf("\n");

	//shift around
	printf("shift around \n");
	cudaMemset(d_out, 0, byte_size);
	test_shfl_shift_around << < grid, block >> > (d_in, d_out, 2);
	cudaDeviceSynchronize();

	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);

	print_array(h_in, size);
	print_array(h_ref, size);
	printf("\n");

	//shuffle xor butterfly
	printf("shuffle xor butterfly \n");
	cudaMemset(d_out, 0, byte_size);
	test_shfl_xor_butterfly << < grid, block >> > (d_in, d_out);
	cudaDeviceSynchronize();

	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);

	print_array(h_in, size);
	print_array(h_ref, size);
	printf("\n");

	cudaFree(d_out);
	cudaFree(d_in);
	free(h_ref);
	free(h_in);

	cudaDeviceReset();
	return EXIT_SUCCESS;
}