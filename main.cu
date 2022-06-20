%%cu
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

#define N 2000
#define DIV 100

// Выполняется на GPU
__global__ void mult_matrix(int* matrix1, int* matrix2, int* result, int i) {
	int k = blockIdx.x * (N / DIV) + threadIdx.x;
	int j = blockIdx.y * (N / DIV) + threadIdx.y;
	result[k * N + j] += matrix1[k * N + i] * matrix2[i * N + j];
}

__host__ void insert_matrix(int* matrix){
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j)
			matrix[i * N + j] = rand() % 10;
	}
}

int main() {

	printf("N = %d, DIV = %d \n", N, DIV);

	//заполнение матриц
	int *matrix1,
		*matrix2,
		*result;

	matrix1 = new int[N * N];
	matrix2 = new int[N * N];
	result = new int[N * N];

	insert_matrix(matrix1);
	insert_matrix(matrix2);
	memset(result, 0, N*N);

	int* dev_matrix1, *dev_matrix2, *dev_result;
	cudaError_t cudaStatus;

	// Выделение памяти на видеокарте
	cudaMalloc((void**)&dev_matrix1, N * N * sizeof(int));
	cudaMalloc((void**)&dev_matrix2, N * N * sizeof(int));
	cudaMalloc((void**)&dev_result, N * N * sizeof(int));

	cudaError_t error;
	// Копирование матриц из оперативной памяти в память видеокарты
	error = cudaMemcpy(dev_matrix1, matrix1, N * N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
		printf("%s\n", cudaGetErrorString(error));

	error = cudaMemcpy(dev_matrix2, matrix2, N * N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
		printf("%s\n", cudaGetErrorString(error));

	error = cudaMemcpy(dev_result, result, N * N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
		printf("%s\n", cudaGetErrorString(error));

	dim3 grid(DIV, DIV);
	dim3 blocks(N / DIV, N / DIV);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	//запускаем алгоритм
	for (int i = 0; i < N; ++i) 
		mult_matrix << <grid, blocks >> > (dev_matrix1, dev_matrix2, dev_result, i);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//записываем время работы
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);

	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("%s\n", cudaGetErrorString(error));


	cudaDeviceSynchronize();

	error = cudaMemcpy(matrix1, dev_matrix1, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
		printf("%s\n", cudaGetErrorString(error));

	error = cudaMemcpy(matrix2, dev_matrix2, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
		printf("%s\n", cudaGetErrorString(error));

	error = cudaMemcpy(result, dev_result, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
		printf("%s\n", cudaGetErrorString(error));

	printf("\nTIME: %fs \n", time/1000);

	delete matrix1;
	delete matrix2;
	delete result;
	cudaFree(dev_matrix1);
	cudaFree(dev_matrix2);
	cudaFree(dev_result);
	return 0;
}