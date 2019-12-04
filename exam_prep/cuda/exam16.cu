#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void test(float* in, float* out, int N) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex = threadIdx.x;

	__shared__ float shared_array[66];

	if (index < N) {
		shared_array[lindex + 1] = in[index];

		if (lindex == 0 && index != 0) {
			shared_array[0] = in[index - 1];
		}

		if (lindex == 63 && index != N - 1) {
			shared_array[65] = in[index + 1];
		}

		__syncthreads();
	}

	if (index == 0 || index == N-1) {
		out[index] = 0;
	}

	else if (index < N) {
		out[index] = (shared_array[lindex] + shared_array[lindex + 1] + shared_array[lindex + 2]) / 3.0f;
	}

}

float* func(int N) {
	size_t size = sizeof(int) * N;
	float* h_in = (float *)malloc(size);
	float* h_out = (float *)malloc(size);

	float* d_in;
	float* d_out;

	cudaMalloc((void **)&d_in, size);
	cudaMalloc((void **)&d_out, size);

	for (int i = 0; i < N; i++) {
		h_in[i] = (float)(i % 3) + 1;
	}

	cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
	int nBlocks = N / 64;
	if (nBlocks % N > 0) {
		nBlocks++;
	}

	test<<<nBlocks, 64>>>(d_in, d_out, N);
	cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

	return h_out;
};

int main(int argc, char **argv) {
	int size = 70;
	float* res = func(size);
	printf("[");
	for (int i = 0; i < size-1; i++) {
		printf("%f, ", res[i]);
	}
	printf("%f]\n", res[size-1]);
};

