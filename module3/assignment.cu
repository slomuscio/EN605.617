//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

void cpuAlgorithm(float *input, float *output, int N)
{
    for (int i = 0; i < N; i++)
    {
        output[i] = input[i] * input[i] + 2.0f * input[i];
    }
}

// CPU algorithm with conditional branching
void cpuBranchAlgorithm(float *input, float *output, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (i % 2 == 0)
        {
            output[i] = input[i] * input[i] + 2.0f * input[i];
        }
        else
        {
            output[i] = input[i] * input[i] - 2.0f * input[i];
        }
    }
}

__global__ void simpleKernel(float *input, float *output, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        output[i] = input[i] * input[i] + 2.0f * input[i];
    }
}


__global__ void branchKernel(float *input, float *output, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        if (i % 2 == 0)
        {
            output[i] = input[i] * input[i] + 2.0f * input[i];
        }
        else
        {
            output[i] = input[i] * input[i] - 2.0f * input[i];
        }
    }
}






int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;
	
	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	printf("Total elements: %d\n", totalThreads);
    	printf("Threads per block: %d\n", blockSize);
    	printf("Number of blocks: %d\n", numBlocks);
    
	// Allocate host memory
    	float *h_input = (float*)malloc(totalThreads * sizeof(float));
	float *h_cpu_output = (float*)malloc(totalThreads * sizeof(float));
	float *h_gpu_output = (float*)malloc(totalThreads * sizeof(float));
    	//float *h_output = (float*)malloc(totalThreads * sizeof(float));

    	// Initialize input
    	for (int i = 0; i < totalThreads; i++)
    	{
    	    h_input[i] = (float)i;
    	}

	// CPU
	auto cpu_start = std::chrono::high_resolution_clock::now();
	cpuAlgorithm(h_input, h_cpu_output, totalThreads);
	auto cpu_end = std::chrono::high_resolution_clock::now();
	double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

	auto cpu_branch_start = std::chrono::high_resolution_clock::now();
	cpuBranchAlgorithm(h_input, h_cpu_output, totalThreads);
	auto cpu_branch_end = std::chrono::high_resolution_clock::now();
	double cpu_branch_time = std::chrono::duration<double, std::milli>(cpu_branch_end - cpu_branch_start).count();

	// GPU 
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);

	cudaEvent_t gpu_branch_start, gpu_branch_stop;
	cudaEventCreate(&gpu_branch_start);
	cudaEventCreate(&gpu_branch_stop);

	float *d_input;
    	float *d_output;

	cudaMalloc((void**)&d_input, totalThreads * sizeof(float));
    	cudaMalloc((void**)&d_output, totalThreads * sizeof(float));

    	cudaMemcpy(d_input, h_input, totalThreads * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventRecord(gpu_start);
	simpleKernel<<<numBlocks, blockSize>>>(d_input, d_output, totalThreads);
	cudaEventRecord(gpu_stop);
	
	cudaEventRecord(gpu_branch_start);
	branchKernel<<<numBlocks, blockSize>>>(d_input,d_output,totalThreads);
	cudaEventRecord(gpu_branch_stop);

	cudaDeviceSynchronize();

	float gpu_time = 0.0f;
	float gpu_branch_time = 0.0f;
	
	cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
	cudaEventElapsedTime(&gpu_branch_time, gpu_branch_start, gpu_branch_stop);
	printf("\nTiming Results:\n");
	printf("CPU no branch time: %.6f ms\n", cpu_time);
	printf("GPU no branch kernel time: %.6f ms\n", gpu_time);
	printf("CPU branch time: %.6f ms\n", cpu_branch_time);
	printf("GPU branch kernel time: %.6f ms\n", gpu_branch_time);

	cudaMemcpy(h_gpu_output, d_output, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);






	// Compare CPU and GPU results
	// bool resultsMatch = true;
	// 
	// for (int i = 0; i < totalThreads; i++)
	// {
	//     if (h_cpu_output[i] != h_gpu_output[i])
	//     {
	//         resultsMatch = false;
	//         printf(
	//             "Mismatch at index %d: CPU = %f, GPU = %f\n",
	//             i,
	//             h_cpu_output[i],
	//             h_gpu_output[i]
	//         );
	//         break;
	//     }
	// }
	// 
	// if (resultsMatch)
	// {
	//     printf("\nCPU and GPU results match!\n");
	// }










	// Check results
	printf("\nResults:\n");
	
	for (int i = 0; i < 5; i++)
	{
	    printf(
	        "input[%d] = %.2f, CPU output = %.2f, GPU output = %.2f\n",
	        i,
	        h_input[i],
	        h_cpu_output[i],
		h_gpu_output[i]
	    );
	}
	
	// Free memory
	cudaFree(d_input);
	cudaFree(d_output);

	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);
	cudaEventDestroy(gpu_branch_start);
        cudaEventDestroy(gpu_branch_stop);
	
	free(h_input);
	free(h_cpu_output);
	free(h_gpu_output);
	
	return 0;
	
}
