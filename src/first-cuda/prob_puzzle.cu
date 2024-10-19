#include <iostream>
#include <string>
#include <stdio.h>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include "device_launch_parameters.h"

#define DICE_MAX 6
#define DICE_MIN 1
#define NUMBER_OF_DICE 1


__global__ void setup_kernel(curandState* state){
int index = threadIdx.x + blockDim.x * blockIdx.x;
curand_init((unsigned long long)clock() + index, index, 0, &state[index]);
}

__global__ void monte_carlo_kernel(curandState* state, unsigned long long int* count, int m, int runs_per_thread) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    __shared__ int cache;

    if (index == 0) {
        cache = 0;
    }
    __syncthreads();

    
    unsigned long long int local_count = 0;

    unsigned int temp = 0;
    unsigned long long int sumroll = 0;

    for (unsigned long long int i = 0; i < runs_per_thread; i++) {
    sumroll = 0;
    temp = 0;
    while (temp < m) {
        sumroll += ceilf(curand_uniform(&state[index]) * DICE_MAX);
        temp++;
    }

    if(sumroll == 3*m) {
        local_count++;
    }
    }

    atomicAdd(&cache, local_count);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, cache);
    }

}

int main() {
    unsigned long long int m = NUMBER_OF_DICE;

    unsigned long long int* h_count;
    unsigned long long int* d_count;

    unsigned long long int BLOCKS = 128;
    unsigned long long int THREADS = 256;
    unsigned long long int RUNS_PER_THREAD = 500000;
    unsigned long long int n = BLOCKS * THREADS * RUNS_PER_THREAD;

    curandState* d_state;
    float chance;

    h_count = (unsigned long long int*)malloc(sizeof(unsigned long long int));
    cudaMalloc((void**)&d_count, sizeof(unsigned long long int));
    cudaMalloc((void**)&d_state, BLOCKS * THREADS *sizeof(curandState));
    cudaMemset(d_count, 0, sizeof(unsigned long long int));



    setup_kernel<<<BLOCKS, THREADS>>>(d_state);

    monte_carlo_kernel<<<BLOCKS,THREADS>>>(d_state, d_count, m, RUNS_PER_THREAD);

    cudaMemcpy(h_count, d_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << *h_count << std::endl;

    chance = float(*h_count) / float(n);

    std::cout << "Chance is " << chance << std::endl;

}