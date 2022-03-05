#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include "ped_model.h"

// constexpr int BLOCK_NUMBER = 4;
// constexpr int BLOCK_SIZE = 256;
constexpr int WEIGHTSUM = 273;

__constant__ int dW[5 * 5];

namespace Ped {

__global__ void initHeatmap(int* hm, int** heatmap) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    heatmap[tid] = hm + SIZE * tid;
    // printf("initHeatmap\n");
}

__global__ void initScaledHeatmap(int* shm, int** scaled_heatmap) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    scaled_heatmap[tid] = shm + SCALED_SIZE * tid;
    // printf("initScaledHeatmap\n");
}

__global__ void initBlurredHeatmap(int* bhm, int** blurred_heatmap) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    blurred_heatmap[tid] = bhm + SCALED_SIZE * tid;
    // printf("initBlurredHeatmap\n");
}

__global__ void heatFades(int** heatmap) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int k = 0; k < SIZE; k++) {
        // heat fades
        heatmap[k][tid] = (int)round(heatmap[k][tid] * 0.80);
    }
    // printf("heatFades\n");
}

__global__ void countHeatmap(int** heatmap,
                             float* desiredXs,
                             float* desiredYs,
                             const int agentSize) {
    /*
        Count how many agents want to go to each location
    */
    // printf("-S countHeatmap\n");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // boundary check
    if (tid < agentSize) {
        // printf("-IFS countHeatmap\n");
        int x = (int)desiredXs[tid];
        int y = (int)desiredYs[tid];
        printf("x=%d, y=%d\n", x, y);
        if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
            return;
        atomicAdd(&heatmap[y][x], 40);
        printf("-IFE countHeatmap\n");
    }
      
   //  printf("-E countHeatmap\n");
}

__global__ void colorHeatmap(int** heatmap,
                             float* desiredXs,
                             float* desiredYs, 
                             const int agentSize) {
    printf("-s colorHeatmap\n");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < agentSize) {
        int x = (int)desiredXs[tid];
        int y = (int)desiredYs[tid];

        atomicMin(&heatmap[y][x], 255);
    }
    printf("-E colorHeatmap\n");
}

__global__ void scaleHeatmap(int** heatmap, int** scaled_heatmap) {
    /*
        Scale the data for visual representation
    */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int x = 0; x < SIZE; x++) {
        int value = heatmap[tid][x];
        for (int cellY = 0; cellY < CELLSIZE; cellY++) {
            for (int cellX = 0; cellX < CELLSIZE; cellX++) {
                scaled_heatmap[tid * CELLSIZE + cellY][x * CELLSIZE + cellX] =
                    value;
            }
        }
    }

    // printf("scaleHeatmap\n");
}

__global__ void filterHeatmap(int** scaled_heatmap,
                              int** blurred_heatmap,
                              const int w[5][5]) {
    /*
        Apply gaussian blur filter
    */
   printf("filterHeatmap starts\n");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= 2 && tid < SCALED_SIZE - 2) {
        for (int j = 2; j < SCALED_SIZE - 2; j++) {
            int sum = 0;
            for (int k = -2; k < 3; k++) {
                for (int l = -2; l < 3; l++) {
                    sum += w[2 + k][2 + l] * scaled_heatmap[tid + k][j + l];
                }
            }
            int value = sum / WEIGHTSUM;
            blurred_heatmap[tid][j] = 0x00FF0000 | value << 24;
        }
    }
    printf("filterHeatmap\n");
}

void Model::setupHeatmapCUDA() {
    printf("setupHeatmapCUDA Start\n");
    int *hm, *shm, *bhm;

    cudaMalloc(&hm, SIZE * SIZE * sizeof(int));
    cudaMemset(hm, 0, SIZE * SIZE);

    cudaMalloc(&shm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMalloc(&heatmap, SIZE * sizeof(int*));

    cudaMalloc(&scaled_heatmap, SCALED_SIZE * sizeof(int*));

    // blurred_heatmap shouldn't be in device
    cudaMallocHost(&bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMallocHost(&blurred_heatmap, SCALED_SIZE * sizeof(int*));

    cudaMalloc(&desiredXs, agents.size() * sizeof(float));
    cudaMalloc(&desiredYs, agents.size() * sizeof(float));

    initHeatmap<<<1, SIZE>>>(hm, heatmap);
    cudaDeviceSynchronize();

    initScaledHeatmap<<<CELLSIZE, SIZE>>>(shm, scaled_heatmap);
    cudaDeviceSynchronize();

    initBlurredHeatmap<<<CELLSIZE, SIZE>>>(bhm, blurred_heatmap);
    cudaDeviceSynchronize();

    for (int i = 0; i < SCALED_SIZE; i++) {
        blurred_heatmap[i] = bhm + SCALED_SIZE * i;
    }

    const int w[5][5] = {{1, 4, 7, 4, 1},
                         {4, 16, 26, 16, 4},
                         {7, 26, 41, 26, 7},
                         {4, 16, 26, 16, 4},
                         {1, 4, 7, 4, 1}};
    cudaMemcpyToSymbol(dW, w, 5 * 5 * sizeof(int));


    printf("setupHeatmapCUDA Ends\n");
}

void Model::updateHeatmapCUDA() {
    printf("--- updateHeatmapCUDA starts\n");

    // init stream
    cudaStream_t fadeStream;
    cudaStream_t countStream;
    cudaStream_t otherStream;

    cudaEvent_t fadeFinish;
    cudaEvent_t countFinish;

    cudaStreamCreate(&fadeStream);
    cudaStreamCreate(&countStream);

    cudaStreamCreate(&otherStream);

    cudaEventCreate(&fadeFinish);
    cudaEventCreate(&countFinish);

    // heatmap fades
    heatFades<<<1, SIZE, 0, fadeStream>>>(heatmap);
    cudaEventRecord(fadeFinish, fadeStream);

    cudaMemcpyAsync(desiredXs, agentSOA->xs,
                    agents.size() * sizeof(float), cudaMemcpyHostToDevice,
                    countStream);
    cudaMemcpyAsync(desiredYs, agentSOA->ys,
                    agents.size() * sizeof(float), cudaMemcpyHostToDevice,
                    countStream);
    
    cudaStreamWaitEvent(countStream, fadeFinish);
    printf("--- updateHeatmapCUDA copy ends\n");

    countHeatmap<<<CELLSIZE, SIZE, 0, countStream>>>(heatmap, desiredXs,
                                                     desiredYs, agents.size());
    cudaEventRecord(countFinish, countStream);

    cudaStreamWaitEvent(otherStream, countFinish);

    printf("-- color starts\n");
    // Color Heatmap
    colorHeatmap<<<1, SIZE, 0, otherStream>>>(heatmap, desiredXs, desiredYs, agents.size());

    // Scale Heatmap
    scaleHeatmap<<<1, SIZE, 0, otherStream>>>(heatmap, scaled_heatmap);

    // Apply gaussian blur filter
    constexpr int w[5][5] = {{1, 4, 7, 4, 1},
                             {4, 16, 26, 16, 4},
                             {7, 26, 41, 26, 7},
                             {4, 16, 26, 16, 4},
                             {1, 4, 7, 4, 1}};

    filterHeatmap<<<1, SIZE, 0, otherStream>>>(scaled_heatmap, blurred_heatmap,
                                               w);

    cudaStreamSynchronize(otherStream);

    // destory events and streams
    cudaStreamDestroy(fadeStream);
    cudaStreamDestroy(countStream);
    cudaStreamDestroy(otherStream);

    cudaEventDestroy(fadeFinish);
    cudaEventDestroy(countFinish);

    cudaStreamDestroy(fadeStream);
    cudaStreamDestroy(fadeStream);
    cudaStreamDestroy(fadeStream);
    printf("--- updateHeatmapCUDA ends\n");
}
}  // namespace Ped