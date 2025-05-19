#include "scan.cuh"

#include "common.h"

#define BLOCK_SIZE 512

// cpu implementation
void inclusive_scan_cpu(int *input, int *output, int size)
{
	output[0] = input[0];

	// make sense, just do prefix sum including index's element
	for (int i = 1; i < size; i++)
	{
		output[i] = output[i - 1] + input[i];
	}
}

void exclusive_scan_cpu(int *input, int *output, int size)
{
	output[0] = 0;

	// make sense, just do prefix sum including index's element
	for (int i = 1; i < size; i++)
	{
		output[i] = output[i - 1] + input[i-1];
	}
}

// naive idea implementation
__global__ void naive_inclusive_scan_single_block(int *input, int size)
{
	int tid = threadIdx.x;
	// only support 1 thread block, so gid should be identical to tid
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
	{
		// only first several elements contain final result
		// since tid == gid, so gid-stride should equal or larger than 0
		for (int stride = 1; stride <= tid; stride *= 2)
		{
			input[gid] += input[gid - stride];
			// make sure to sync in each iteration
			__syncthreads();
		}
	}
}

__global__ void efficient_inclusive_scan_single_block(int *input,int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
	{
		for (int stride = 1; stride <= tid; stride *= 2)
		{
			input[gid] += input[gid - stride];
			__syncthreads();
		}
	}
}

__global__ void efficient_exclusive_scan_single_block(int *input,int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
	{
		// upper-sweep phase (reduction)
		for (int stride = 1; stride < blockDim.x; stride *= 2)
		{
			// this time, do sum for odd indices
			// tid+1 is to make sure -1 won't cause less than 0 index
			// multiple by 2 is to make sure 2n-1
			int index = ( tid + 1 ) * 2 * stride - 1;
			if (index < blockDim.x) {
				input[index] += input[index - stride];
			}
			__syncthreads();
		}

		//set root value to 0
		if (tid == 0)
			input[blockDim.x - 1] = 0;
		
		int temp = 0;

		// down-sweep phase
		for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
		{
			// same as above
			int index = ( tid + 1 ) * 2 * stride - 1;
			if (index < blockDim.x) {
				temp = input[index - stride]; // assign left child to temp
				input[index - stride] = input[index]; // assign left child with parent node's value
				input[index] += temp; // sum left child's original value with parent node's value and assign to right child
			}
			__syncthreads();
		} 
	}
}


__global__ void efficient_inclusive_scan_single_block(int *input,int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
	{
		// upper-sweep phase (reduction)
		for (int stride = 1; stride < blockDim.x; stride *= 2)
		{
			// this time, do sum for odd indices
			// tid+1 is to make sure -1 won't cause less than 0 index
			// multiple by 2 is to make sure 2n-1
			int index = ( tid + 1 ) * 2 * stride - 1;
			if (index < blockDim.x) {
				input[index] += input[index - stride];
			}
			__syncthreads();
		}

		//set root value to 0
		if (tid == 0)
			input[blockDim.x - 1] = 0;
		
		int temp = 0;

		// down-sweep phase
		for (int stride = blockDim.x / 4; stride > 0; stride /= 2)
		{
			// same as above
			int index = ( tid + 1 ) * 2 * stride - 1;
			if (index + stride < blockDim.x) {
				// use intermediate result to calculcate prefix sum
				input[index + stride] += input[index];	
			}
			__syncthreads();
		} 
	}
}


__global__ void sum_aux_values(int *input,  int *aux, int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
	{
		for (int i = 0; i < blockIdx.x; i++)
		{
			input[gid] += aux[i];
			__syncthreads();
		}
	}
}

int main(int argc, char**argv)
{
	printf("Scan algorithm execution starterd \n");

	int input_size = 1 << 10; // 1024
	
	if (argc > 1)
	{
		input_size = 1 << atoi(argv[1]);
	}
	
	const int byte_size = sizeof(int) * input_size;

	int * h_input, *h_output, *h_ref, *h_aux;

	clock_t cpu_start, cpu_end, gpu_start, gpu_end;

	h_input = (int*)malloc(byte_size);
	h_output = (int*)malloc(byte_size);
	h_ref = (int*)malloc(byte_size);

	initialize(h_input, input_size, INIT_ONE);

	cpu_start = clock();
	// inclusive_scan_cpu(h_input, h_output, input_size);
	exclusive_scan_cpu(h_input, h_output, input_size);
	cpu_end = clock();

	int *d_input, *d_aux;
	cudaMalloc((void**)&d_input, byte_size);

	cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

	dim3 block(BLOCK_SIZE); // 512
	dim3 grid(input_size/ block.x); // by default is 2

	int aux_byte_size = block.x * sizeof(int);
	cudaMalloc((void**)&d_aux , aux_byte_size);

	h_aux = (int*)malloc(aux_byte_size);
	
	// this is incorrect because should only use 1 thread block in naive implementation
	// naive_inclusive_scan_single_block << <grid, block >> > (d_input, input_size);
	// naive_inclusive_scan_single_block << <1, 1024 >> > (d_input, input_size);
	// efficient_exclusive_scan_single_block << <1, 1024 >> > (d_input, input_size);
	efficient_inclusive_scan_single_block << <1, 1024 >> > (d_input, input_size);
	cudaDeviceSynchronize();

	cudaMemcpy(h_aux, d_aux, aux_byte_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost);

	print_arrays_toafile(h_ref, input_size, "input_array.txt");

	for (int i = 0; i < input_size; i++)
	{
		for (int j = 0; j < i / BLOCK_SIZE ; j++)
		{
			h_ref[i] += h_aux[j];
		}
	}

	print_arrays_toafile(h_aux,grid.x, "aux_array.txt");

	//sum_aux_values << < grid, block >> > (d_input, d_aux, input_size);
	//cudaDeviceSynchronize();

	//cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost );
	//print_arrays_toafile_side_by_side(h_ref, h_output, input_size, "scan_outputs.txt");

	compare_arrays(h_ref, h_output, input_size);

	gpuErrchk(cudaDeviceReset());
	return 0;
}