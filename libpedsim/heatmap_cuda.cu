#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "ped_model.h"
#include <vector>

constexpr int BLOCK_NUMBER = 4;
constexpr int BLOCK_SIZE = 256;
constexpr int WEIGHTSUM = 273;

namespace Ped {

    __global__ void initDesiredXY(float* desiredXs, float* desiredYs, TagentSOA* agentSOA) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        desiredXs[tid] = agentSOA->desiredXs[tid];
        desiredYs[tid] = agentSOA->desiredXs[tid];
        printf("initDesiredXY\n");
    }

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
            heatmap[tid][k] = (int)round(heatmap[tid][k] * 0.80);
        }
        printf("heatFades\n");
    }

    __global__ void countHeatmap(int** heatmap, float* desiredXs, float* desiredYs, const int agentSize) {
        /*
            Count how many agents want to go to each location
        */
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int x = desiredXs[tid];
        int y = desiredYs[tid];

        if (tid > agentSize)
            return;

        if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
            return;

        // intensify heat for better color results
        atomicAdd(&heatmap[y][x], 40);

        printf("countHeatmap\n");
    }

    __global__ void colorHeatmap(int** heatmap, float* desiredXs, float* desiredYs) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int x = desiredXs[tid];
        int y = desiredYs[tid];
        for (int k = 0; k < SIZE; k++) {
            atomicMin(&heatmap[y][x], 255);
        }
        printf("colorHeatmap\n");
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

        printf("scaleHeatmap\n");
    }

    __global__ void filterHeatmap(int** scaled_heatmap, int** blurred_heatmap, const int w[5][5]) {
        /*
            Apply gaussian blur filter
        */
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
        int *hm, *shm, *bhm;

        cudaMallocManaged(&hm, SIZE * SIZE * sizeof(int));
        cudaMemset(hm, 0, SIZE * SIZE);

        cudaMallocManaged(&shm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
        cudaMallocManaged(&bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int));

        cudaMallocManaged(&heatmap, SIZE * sizeof(int*));

        cudaMallocManaged(&scaled_heatmap, SCALED_SIZE * sizeof(int*));
        cudaMallocManaged(&blurred_heatmap, SCALED_SIZE * sizeof(int*));

        int agentSize = agents.size();
        cudaMallocManaged(&desiredXs, agents.size() * sizeof(int));
        cudaMallocManaged(&desiredYs, agents.size() * sizeof(int));

        initHeatmap<<<1, SIZE>>>(hm, heatmap);
        cudaDeviceSynchronize();

        initScaledHeatmap<<<CELLSIZE, SIZE>>>(shm, scaled_heatmap);
        cudaDeviceSynchronize();

        initBlurredHeatmap<<<CELLSIZE, SIZE>>>(bhm, blurred_heatmap);
        cudaDeviceSynchronize();

        initDesiredXY<<<1, agentSize>>>(desiredXs, desiredYs, agentSOA);
        cudaDeviceSynchronize();
    }

    void Model::updateHeatmapCUDA() {
        float time1, time2, time3, time4;
        
        // Heatfades
        cudaEvent_t fadeBegin, fadeEnd;
        cudaEventCreate(&fadeBegin);
        cudaEventCreate(&fadeEnd);
        cudaEventRecord(fadeBegin, 0);

        heatFades<<<1, SIZE>>>(heatmap);

        cudaEventRecord(fadeEnd, 0);
        cudaEventSynchronize(fadeEnd);
        cudaEventElapsedTime(&time1, fadeBegin, fadeEnd);
        cudaEventDestroy(fadeBegin);
        cudaEventDestroy(fadeEnd);


        // Heatmap Count 
        int agentSize = agents.size();
        cudaEvent_t countBegin, countEnd;
        cudaEventCreate(&countBegin);
        cudaEventCreate(&countEnd);
        cudaEventRecord(countBegin, 0);

        countHeatmap<<<CELLSIZE, SIZE>>>(heatmap, desiredXs, desiredYs, agentSize);

        cudaEventRecord(countEnd, 0);
        cudaEventSynchronize(countEnd);
        cudaEventElapsedTime(&time2, countBegin, countEnd);
        cudaEventDestroy(countBegin);
        cudaEventDestroy(countEnd);

        // Color Heatmap
        colorHeatmap<<<1, SIZE>>>(heatmap, desiredXs, desiredYs);
        cudaDeviceSynchronize();

        // Scale Heatmap
        cudaEvent_t scaleBegin, scaleEnd;
        cudaEventCreate(&scaleBegin);
        cudaEventCreate(&scaleEnd);
        cudaEventRecord(scaleBegin, 0);

        scaleHeatmap<<<1,SIZE>>>(heatmap, scaled_heatmap);

        cudaEventRecord(scaleEnd, 0);
        cudaEventSynchronize(scaleEnd);
        cudaEventElapsedTime(&time3, scaleBegin, scaleEnd);
        cudaEventDestroy(scaleBegin);
        cudaEventDestroy(scaleEnd);

        // Apply gaussian blur filter
        constexpr int w[5][5] = {{1, 4, 7, 4, 1},
                         {4, 16, 26, 16, 4},
                         {7, 26, 41, 26, 7},
                         {4, 16, 26, 16, 4},
                         {1, 4, 7, 4, 1}};

        cudaEvent_t filterBegin, filterEnd;
        cudaEventCreate(&filterBegin);
        cudaEventCreate(&filterEnd);
        cudaEventRecord(filterBegin, 0);

        filterHeatmap<<<1,SIZE>>>(scaled_heatmap, blurred_heatmap, w);

        cudaEventRecord(filterEnd, 0);
        cudaEventSynchronize(filterEnd);
        cudaEventElapsedTime(&time4, filterBegin, filterEnd);
        cudaEventDestroy(filterBegin);
        cudaEventDestroy(filterEnd);

    }

}  // namespace Ped