#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

void CHECK(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		printf("Error : %s : %d, ", __FILE__, __LINE__);
		printf("code : %d, reason: %s \n", error, cudaGetErrorString(error));
		exit(1);
	}
    printf("No error found \n");
}

void checkResult(float *host_ref, float *gpu_ref, const int N)
{
	double epsilon = 0.0000001;
	bool match = 1;

	for (size_t i = 0; i < N; i++)
	{
		if (abs(host_ref[i] - gpu_ref[i]) > epsilon)
		{
			match = 0;
			printf("Arrays do not match! \n");
			printf("host %5.2f  gpu %5.2f at current %d\n", host_ref[i], gpu_ref[i], N);
			break;
		}
	}

	if (match) printf("Arrays match . \n\n");
}

void initialize_data_s(float * ip, int size)
{
	time_t t;
	srand((unsigned)time(&t));

    // initialize random float value
	for (size_t i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

// used for comparison
void sum_array_cpu(float * a, float *  b, float * c, const int N)
{
	for (size_t i = 0; i < N; i++)
	{
		c[i] = a[i] + b[i];
	}
}

// TODO: extend more than 1 block in grid
__global__ void sum_array_gpu(float * a, float *  b, float * c, int size)
{
	// int i = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        c[gid] = a[gid] + b[gid];
    }
	// printf("a =%f  b = %f c = %f \n", a[i], b[i], c[i]);
}

void run_code()
{
    cudaError error;

	int element_Count = 1 << 25;
    int block_size = 128;
    printf("element count %d \n", element_Count);
	size_t number_bytes = element_Count * sizeof(float);

	float *h_a, *h_b, *host_ref, *gpu_ref;

    // c programming
	h_a = (float *)malloc(number_bytes);
	h_b = (float *)malloc(number_bytes);
	host_ref = (float *)malloc(number_bytes);
	gpu_ref = (float *)malloc(number_bytes);

	initialize_data_s(h_a, element_Count);
	initialize_data_s(h_b, element_Count);

	memset(host_ref, 0, number_bytes);
	memset(gpu_ref, 0, number_bytes);

	float *d_a, *d_b, *d_c;
	error = cudaMalloc((float **)&d_a, number_bytes);
    CHECK(error);
	cudaMalloc((float **)&d_b, number_bytes);
	cudaMalloc((float **)&d_c, number_bytes);

    clock_t htod_start, htod_end;
    htod_start = clock();
	cudaMemcpy(d_a, h_a, number_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, number_bytes, cudaMemcpyHostToDevice);
    htod_end = clock();

    // here only use 1 grid
    // for block, it has 32 threads in x dim
	dim3 block(block_size);
	dim3 grid((element_Count / block.x)+1); // +1 make sure threads are enough

    // therefore, no need to calculate specific global id here
    clock_t gpu_start, gpu_end;
    gpu_start = clock();
	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, element_Count);
    // seems that no use of following line also works
    cudaDeviceSynchronize();
    gpu_end = clock();

    // this is because when calling cudaMemcpy, it implicitly sync
    clock_t dtoh_start, dtoh_end;
    dtoh_start = clock();
	cudaMemcpy(gpu_ref, d_c, number_bytes, cudaMemcpyDeviceToHost);
    dtoh_end = clock();

    // finding: kernel computation time is very small, memory copy is bottleneck
    printf("host to device time: %4.6f \n", (double)((double)(htod_end-htod_start)/CLOCKS_PER_SEC));
    printf("kernel time: %4.6f \n", (double)((double)(gpu_end-gpu_start)/CLOCKS_PER_SEC));
    printf("device to host time: %4.6f \n", (double)((double)(dtoh_end-dtoh_start)/CLOCKS_PER_SEC));

    // can be used to compare result calculated by cpu
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    sum_array_cpu(h_a, h_b, host_ref, element_Count);
    cpu_end = clock();
    printf("cpu time: %4.6f \n", (double)((double)(cpu_end-cpu_start)/CLOCKS_PER_SEC));
    checkResult(host_ref, gpu_ref, element_Count);

    // free memory in device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

    // free memory in host
	free(h_a);
	free(h_b);
	free(host_ref);
	free(gpu_ref);
}

int main()
{
	run_code();
	system("pause");
	return 0;
}