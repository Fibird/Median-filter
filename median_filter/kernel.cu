#include "cuda_runtime.h"
#include <stdio.h>
#include <memory.h>
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>

#define threadsPerBlock 256
#define RADIUS 2

// Signal/image element type
typedef unsigned char element;
//   1D MEDIAN FILTER implementation
//     signal - input signal
//     result - output signal
//     N      - length of the signal


__global__ void _medianfilter(const element* signal, element* result)
{
	__shared__ element cache[threadsPerBlock + 2 * RADIUS];
	element window[5];
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
	for (int j = 0; j < 2 * RADIUS + 1; ++j)
		window[j] = cache[threadIdx.x + j];
	// Orders elements (only half of them)
	for (int j = 0; j < RADIUS + 1; ++j)
	{
		// Finds position of minimum element
		int min = j;
		for (int k = j + 1; k < 2 * RADIUS + 1; ++k)
			if (window[k] < window[min])
				min = k;
		// Puts found minimum element in its place
		const element temp = window[j];
		window[j] = window[min];
		window[min] = temp;
	}
	// Gets result - the middle element
	result[gindex] = window[RADIUS];
}

//   1D MEDIAN FILTER wrapper
//     signal - input signal
//     result - output signal
//     N      - length of the signal
void medianfilter(element* signal, element* result, int N)
{
	element *dev_extension, *dev_result;

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
	for (int i = 0; i < RADIUS; ++i)
	{
		extension[i] = signal[1 - i];
		extension[N + RADIUS + i] = signal[N - 1 - i];
	}

	cudaMalloc((void**)&dev_extension, (N + 2 * RADIUS) * sizeof(int));
	cudaMalloc((void**)&dev_result, N * sizeof(int));

	// Copies signal to device
	cudaMemcpy(dev_extension, extension, (N + 2 * RADIUS) * sizeof(element), cudaMemcpyHostToDevice);
	//   Call median filter implementation
	_medianfilter<<<blocksPerGrid, threadsPerBlock>>>(dev_extension + RADIUS, dev_result);
	// Copies result to host
	cudaMemcpy(result, dev_result, N * sizeof(element), cudaMemcpyDeviceToHost);

	// Free memory
	free(extension);
	cudaFree(dev_extension);
	cudaFree(dev_result);
}

int main()
{
	IplImage *ImgSrc = cvLoadImage("sample_corrupted.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage *ImgReal = cvLoadImage("sample.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage *ImgDst_CPU = cvCreateImage(cvGetSize(ImgSrc), IPL_DEPTH_8U, 1);
	int Size = ImgSrc->width * ImgSrc->height;
	unsigned char *pSrcData = (unsigned char*)(ImgSrc->imageData);
	unsigned char *pDstData = (unsigned char*)(ImgDst_CPU->imageData);
	int blocksPerGrid = (Size + threadsPerBlock - 1) / threadsPerBlock;

	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	medianfilter<<<blocksPerGrid, threadsPerBlock>>>(pSrcData, pDstData, Size);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%lf.3 ms\n", elapsedTime);

	return 0;
}