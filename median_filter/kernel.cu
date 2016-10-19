#include <stdio.h>
#include <memory.h>
#include <time.h>

#define N 33 * 1024
#define threadsPerBlock 256
#define blocksPerGrid (N + threadsPerBlock - 1) / threadsPerBlock
#define RADIUS 2
#define TIMES 10		// The time of calling _medianfilter
// Signal/image element type
typedef int element;
//   1D MEDIAN FILTER implementation
//     signal - input signal
//     result - output signal
//     L      - length of the signal
void _medianfilter(const element* signal, element* result, int L)
{
	//   Move window through all elements of the signal
	for (int i = 2; i < L - RADIUS; ++i)
	{
		//   Pick up window elements
		element window[2 * RADIUS + 1];
		for (int j = 0; j < 2 * RADIUS + 1; ++j)
			window[j] = signal[i - RADIUS + j];
		//   Order elements (only half of them)
		for (int j = 0; j < RADIUS + 1; ++j)
		{
			//   Find position of minimum element
			int min = j;
			for (int k = j + 1; k < 2 * RADIUS + 1; ++k)
				if (window[k] < window[min])
					min = k;
			//   Put found minimum element in its place
			const element temp = window[j];
			window[j] = window[min];
			window[min] = temp;
		}
		//   Get result - the middle element
		result[i - RADIUS] = window[RADIUS];
	}
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
	element* extension = new element[N + 2 * RADIUS];
	//   Check memory allocation
	if (!extension)
		return;
	//   Create signal extension
	memcpy(extension + RADIUS, signal, N * sizeof(element));
	for (int i = 0; i < RADIUS; ++i)
	{
		extension[i] = signal[1 - i];
		extension[N + RADIUS + i] = signal[N - 1 - i];
	}
	//   Call median filter implementation
	for (int i = 0; i < TIMES; ++i)
		_medianfilter(extension, result ? result : signal, N + 2 * RADIUS);
	//   Free memory
	delete[] extension;
}

int main()
{
	element *Signal, *result;
	float elapsedTime;
	clock_t start, stop;

	FILE *fp;
	
	Signal = (int *)malloc(N * sizeof(int));
	result = (element *)malloc(N * sizeof(element));
	
	for (int i = 0; i < N; i++)
	{
		Signal[i] = i % 5 + 1;
	}
	start = clock();
	medianfilter(Signal, result);
	stop = clock();
	
	elapsedTime = 1000 * ((float) (stop - start)) / CLOCKS_PER_SEC;
	
	printf("%.3lf ms\n", elapsedTime);

	fp = fopen("result.txt", "w");
	if (fp == NULL)
		printf("OPEN FILE FAILS!\n");
	for (int i = 0; i < N; i++)
		fprintf(fp, "%d ", result[i]);

	fclose(fp);
	return 0;
}