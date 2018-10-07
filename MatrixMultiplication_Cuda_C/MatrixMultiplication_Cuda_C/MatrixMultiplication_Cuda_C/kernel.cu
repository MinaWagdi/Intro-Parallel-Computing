
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


__global__ void MatrixMultKernel(const int m, const int n, const int k,
	const float *dev_A, const float *dev_B,float *dev_C)			
{
	int iR = blockIdx.y*blockDim.y + threadIdx.y;
	int iC = blockIdx.x*blockDim.x + threadIdx.x;

	if ((iR < m) && (iC < k)) {
		float result = 0;
		for (int i = 0; i < n; ++i) {
			result += dev_A[iR*n + i] * dev_B[iC + i*k];
		}
		dev_C[iR*k + iC] = result;
	}
}
cudaError_t MatrixMultWithCuda(int m, int n, int k, float* C, const float *A, const float *B);

/*
m is the number of rows of Matrix A
n is the number of rows in Matrix B and the number of Columns in Matrix A
k is the number of rows in Matrix B

Finally the result of the multiplcation is in Matrix C which has number rows = m and number of columns = k

Summary:
A mxn
B nxk
C mxk
*/
void MatrixMult_Sequential(int m, int n, int k, float* A, float* B, float* C) {
	for (int row = 0; row < m; ++row) {
		for (int col = 0; col < k; ++col) {
			float sum = 0;
			for (int i = 0; i < n; ++i) {
				float a = A[row*n + i];
				float b = B[col + i*k];
				sum += a*b;
			}
			C[row*k + col] = sum;
		}
	}
}

float* generateArrayWithSize(int x,int y) {
	float* arr;
	for (int i = 0; i < x*y; i++)
	{
		arr[i] = rand() % 250;
	}
	return arr;
}

int main()
{
	int m = 1000, n =1000, k = 1000 ;
	
	float*A= (float*)malloc(m*n * sizeof(float));
	for (int i = 0; i <m&n; i++)
	{
		A[i] = rand() % 50+1;
	}
	
	float*B = (float*)malloc(n*k * sizeof(float));
	for (int i = 0; i <m&n; i++)
	{
		B[i] = rand() % 50 +1;
	}
	
	float*C = (float*)malloc(m*k*sizeof(float));
	
    
	printf("\n\n\nMatrix Multiplication With Cuda\n\n");
	clock_t start_t_Cuda, end_t_Cuda;
	double totaltime_Cuda;
	start_t_Cuda = clock();
    // Add vectors in parallel.
    cudaError_t cudaStatus = MatrixMultWithCuda(m, n, k, C,A, B);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	/*
	printf("Matrix A\n");
	for (int i = 0; i < m*n; i++) {
		printf("%f     ", A[i]);
		if ((i+1)%n == 0) {
			printf("\n");
		}
	}
	printf("Matrix B\n");
	for (int i = 0; i < n*k; i++) {
		printf("%f     ", B[i]);
		if ((i+1)%k == 0) {
			printf("\n");
		}
	}
	
	printf("Matrix C\n");
	for (int i = 0; i < m*k; i++) {
		printf("%f     ", C[i]);
		if ((i + 1) %k == 0) {
			printf("\n");
		}
	}
	*/
	end_t_Cuda = clock();
	totaltime_Cuda = (double)(end_t_Cuda - start_t_Cuda) / CLOCKS_PER_SEC;
	printf("total time: %f seconds\n", totaltime_Cuda);

	printf("\n\n\nMatrix Multiplication using Sequential C Code\n\n\n");
	
	clock_t start_t_seq, end_t_seq;
	double totaltime_seq;
	start_t_seq = clock();

	MatrixMult_Sequential(m, n, k, A, B, C);
	/*
	printf("Matrix A\n");
	for (int i = 0; i < m*n; i++) {
		printf("%f     ", A[i]);
		if ((i + 1) %n == 0) {
			printf("\n");
		}
	}
	printf("Matrix B\n");
	for (int i = 0; i < n*k; i++) {
		printf("%f     ", B[i]);
		if ((i + 1) %k == 0) {
			printf("\n");
		}
	}
	
	printf("Matrix C\n");
	for (int i = 0; i < m*k; i++) {
		printf("%f     ", C[i]);
		if ((i + 1) %k == 0) {
			printf("\n");
		}
	}
	*/
	end_t_seq = clock();
	totaltime_seq = (double)(end_t_seq - start_t_seq) / CLOCKS_PER_SEC;
	printf("total time: %f seconds\n", totaltime_seq);
    
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t MatrixMultWithCuda(int m,int n,int k,float* C, const float *A, const float *B)
{
    float *dev_A = 0;
    float *dev_B = 0;
    float *dev_C = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_C, m*k* sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_A, m*n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_B, n*k * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_A, A, m*n * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_B, B, n*k * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	dim3 dimGrid(64, 64, 1);
	dim3 dimBlock((m-1)/64+1, (k-1)/64+1, 1);

    // Launch a kernel on the GPU with one thread for each element.
	MatrixMultKernel <<<dimGrid, dimBlock>>>(m,n,k, dev_A, dev_B,dev_C);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(C, dev_C, m*k * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_C);
    cudaFree(dev_B);
    cudaFree(dev_A);
    
    return cudaStatus;
}
