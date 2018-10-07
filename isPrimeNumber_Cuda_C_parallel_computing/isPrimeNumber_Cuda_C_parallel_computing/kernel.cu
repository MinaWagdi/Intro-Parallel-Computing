
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include<time.h>
//#include "pch.h"



__global__ void checkPrimeNumberKernel(int* d_result, int* d_number, int n) {
	
	// initialize the variable i so that each thread has a different value of the i 
	// There is many thread blocks, so block index is the index of the block
	// block Dimension is the number of threads found in a block
	// thread index is the index of the thread inside a block
	int num = *d_number;
	int i = threadIdx.x+blockDim.x*blockIdx.x;
	
	if (i>1) {
		if ((*d_number % i) == 0) {
			d_result[i] = 0;
		}
		else {
			d_result[i] = 1;
		}
	}
}

cudaError_t checkPrimeNumber(int* h_result,const int* h_number,int n)
{
	//get the size that I want to allocate
	int size = n * sizeof(int);

	//d_number is the variable used by the GPU
	int* d_result;
	int* d_number;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	/*
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	*/

	//allocate part in the global memory in the GPU with the variable "size"
	cudaStatus = cudaMalloc((void**)&d_number, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_result, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//after allocating part of the general memory in the GPU, copy the number to the GPU
	cudaStatus=cudaMemcpy(d_number, h_number, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	checkPrimeNumberKernel << <ceil(n / 256.0), 256 >> >(d_result,d_number,  n);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	//d_result[0] = 1;
	//d_result[1] = 1;
	cudaStatus=cudaMemcpy(h_result, d_result, n*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	cudaFree(d_number);
	cudaFree(d_result);

	return cudaStatus;
}

bool isPrime_sequential(int num) {
	bool is_prime = true;
	for (int i = 2; i < num/2; i++) {
		if (num % i == 0.0) {
			is_prime = false;
			break;
		}
	}
	return is_prime;
}

int main()
{
	/*NOTE : TO TRY A DIFFERENT VALUE, CHANGE IT IN THE "number" VARIABLE AND IN THE ARRAY "isPrime" */
	bool prime = true;
    int number = 23;
	int size = number / 2;
	int isPrime[23 / 2];
	
	printf("Starting parallel computing\n Checking Number : %d\n", number);
    
	clock_t start_t_parallel, end_t_parallel;
	double totaltime;
	start_t_parallel = clock();

	cudaError_t cudaStatus = checkPrimeNumber(isPrime,&number, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	/*
	for (int i = 2; i<number / 2; i++) {
		printf("%d\n", isPrime[i]);
	}
	*/
	for ( int i =2; i<number/2;i++){
		if (isPrime[i] == 0) {
			printf("NOT PRIME\n");
			prime = false;
			break;
		}
	}
	if (prime) {
		printf("PRIME\n");
	}
	
	end_t_parallel = clock();
	totaltime = (double)(end_t_parallel - start_t_parallel) / CLOCKS_PER_SEC;
	printf("total time of the Parallel code: %f seconds\n", totaltime);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	clock_t start_t_sequential, end_t_sequential;
	double totaltime_seq;
	start_t_sequential = clock();

	printf("CHECKING PRIME NUMBERS USING SEQUENTIAL CODE\n");
	bool isPrimeSeq = isPrime_sequential(number);
	
	if (isPrimeSeq) {
		printf("PRIME NUMBER\n");
	}
	else {
		printf("NOT PRIME NUMBER\n");
	}

	end_t_sequential = clock();
	totaltime_seq = (double)(end_t_sequential - start_t_sequential) / CLOCKS_PER_SEC;
	printf("total time: %f seconds\n", totaltime_seq);
    return 0;
}

