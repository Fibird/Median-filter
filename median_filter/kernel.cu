#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <memory.h>
#include <algorithm>

#define N 33 * 1024
#define threadsPerBlock 256
#define blocksPerGrid (N + threadsPerBlock - 1) / threadsPerBlock
#define RADIUS 2
// Signal/image element type
typedef int element;
//   1D MEDIAN FILTER implementation
//     signal - input signal
//     result - output signal
//     N      - length of the signal


__global__ void _medianfilter(const element* signal, element* result)
{
	element window[5];
	__shared__ element cache[threadsPerBlock + 2 * RADIUS];
	int gindex = threadIdx.x + blockDim.x * blockIdx.x;
	int lindex = threadIdx.x + RADIUS;
	// Reads input elements into shared memory
	cache[lindex] = signal[gindex];
	if (threadIdx.x < RADIUS)
	{
		cache[lindex - RADIUS] = signal[gindex - RADIUS];
		cache[lindex + threadsPerBlock] = signal[gindex + threadsPerBlock];
	}
	__syncthreads();
	for (int j = 0; j < 5; ++j)
		window[j] = cache[threadIdx.x  + j];
	// Orders elements (only half of them)
	for (int j = 0; j < 3; ++j)
	{
		// Finds position of minimum element
		int min = j;
		for (int k = j + 1; k < 5; ++k)
			if (window[k] < window[min])
				min = k;
		// Puts found minimum element in its place
		const element temp = window[j];
		window[j] = window[min];
		window[min] = temp;
	}
	// Gets result - the middle element
	result[gindex] = window[2];
}

//   1D MEDIAN FILTER wrapper
//     signal - input signal
//     result - output signal
//     N      - length of the signal
void medianfilter(element* signal, element* result)
{
	//   Check arguments
	if (!signal || N < 1)
		return;
	//   Treat special case N = 1
	if (N == 1)
	{
		if (result)
			result[0] = signal[0];
		return;
	}
	//   Allocate memory for signal extension
	element* extension = (element*)malloc((N + 2 * RADIUS) * sizeof(element));
	//   Check memory allocation
	if (!extension)
		return;
	//   Create signal extension
	cudaMemcpy(extension + 2, signal, N * sizeof(element), cudaMemcpyHostToHost);
	for (int i = 0; i < 2; ++i)
	{
		extension[i] = signal[1 - i];
		extension[N + 2 + i] = signal[N - 1 - i];
	}

	element *dev_extension, *dev_result;
	dev_extension = (element *)cudaMalloc((void**)&dev_extension, (N + 2 * RADIUS) * sizeof(int));
	dev_result = (int *)cudaMalloc((void**)&dev_result, N * sizeof(int));
	
	// Copies signal to device
	cudaMemcpy(dev_extension, extension, (N + 2 * RADIUS) * sizeof(element), cudaMemcpyHostToDevice);
	//   Call median filter implementation
	_medianfilter<<<blocksPerGrid, threadsPerBlock>>>(dev_extension, dev_result);
	// Copies result to host
	cudaMemcpy(result, dev_result, N * sizeof(element), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i ++)
		printf("%d ", result[i]);
	// Free memory
	free(extension);
	cudaFree(dev_extension);
	cudaFree(dev_result);
}

int main()
{
	int *Signal, *result;
	Signal = (int *)malloc(N * sizeof(int));
	result = (element *)malloc(N * sizeof(element));
	//fill_n(Signal, N, 1);
	for (int i = 0; i < N; i++)
		Signal[i] = 1;
	medianfilter(Signal, result);
	
	return 0;
}