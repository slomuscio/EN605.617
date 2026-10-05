#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__constant__ float constantMultiplier;

// CUDA kernel using global memory
__global__ void globalMemoryKernel(float *input, float *output, int N)
{
    // Shared mem: one float per thread in the block
    extern __shared__ float sharedData[];

    // Register var
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        // Copy from global memory into shared memory
        sharedData[threadIdx.x] = input[i];
    }

    // Make sure that all threads finished writing to the shared memory
    __syncthreads();

    if (i < N)
    {
        // Register vars
        float inputValue = sharedData[threadIdx.x];
        float result = inputValue * constantMultiplier;

        // Write result to global memory
        output[i] = result;
    }
}

int main(int argc, char *argv[])
{
    int totalThreads = 1024;
    int blockSize = 256;

    if (argc >= 2)
    {
        totalThreads = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        blockSize = atoi(argv[2]);
    }

    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    printf("Total elements: %d\n", totalThreads);
    printf("Threads per block: %d\n", blockSize);
    printf("Number of blocks: %d\n", numBlocks);

     // Host memory
    float *h_input = (float*)malloc(totalThreads * sizeof(float));
    float *h_output = (float*)malloc(totalThreads * sizeof(float));

    for (int i = 0; i < totalThreads; i++)
    {
        h_input[i] = (float)i;
    }

     // Global device memory
    float *d_input;
    float *d_output;

    cudaMalloc((void**)&d_input, totalThreads * sizeof(float));
    cudaMalloc((void**)&d_output, totalThreads * sizeof(float));

    // Copy host memory to global device memory
    cudaMemcpy(
        d_input,
        h_input,
        totalThreads * sizeof(float),
        cudaMemcpyHostToDevice
    );

    float multiplier = 2.0f;

    cudaMemcpyToSymbol(
        constantMultiplier,
        &multiplier,
        sizeof(float)
    );

    // Kernel
    globalMemoryKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_input,
        d_output,
        totalThreads
    );

    cudaDeviceSynchronize();

    // Copy global device memory back to host
    cudaMemcpy(
        h_output,
        d_output,
        totalThreads * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    printf("\nResults:\n");

    for (int i = 0; i < 5 && i < totalThreads; i++)
    {
        printf(
            "input[%d] = %.2f, output[%d] = %.2f\n",
            i,
            h_input[i],
            i,
            h_output[i]
        );
    }

    // Clean up stuff
    cudaFree(d_input);
    cudaFree(d_output);

    free(h_input);
    free(h_output);

    return 0;
}