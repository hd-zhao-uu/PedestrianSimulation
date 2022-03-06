#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include "ped_model.h"


constexpr int WEIGHTSUM = 273;

__constant__ int d_w[5 * 5];

namespace Ped {

__global__ void heatFades(int* d_heatmap) {
    long tid = threadIdx.x + blockIdx.x * blockDim.x;
    // heat fades
    d_heatmap[tid] = (int)round(d_heatmap[tid] * 0.80);
    // printf("[DEBUG] In heatFades Done\n");
    
}

__global__ void countHeatmap(int* d_heatmap,
                             float* d_desiredXs,
                             float* d_desiredYs,
                             const int agentSize) {
    /*
        Count how many agents want to go to each location
    */
    printf("-S countHeatmap\n");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // boundary check
    if (tid < agentSize) {
        // printf("-IFS countHeatmap\n");
        int x = (int)d_desiredXs[tid];
        int y = (int)d_desiredYs[tid];
        printf("x=%d, y=%d\n", x, y);
        if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
            return;
        atomicAdd(&d_heatmap[y*SIZE + x], 40);
        printf("-IFE countHeatmap\n");
    }
      
   //  printf("-E countHeatmap\n");
}

__global__ void colorHeatmap(int* d_heatmap,
                             float* d_desiredXs,
                             float* d_desiredYs, 
                             const int agentSize) {
    // printf("-s colorHeatmap\n");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < agentSize) {
        int x = (int)d_desiredXs[tid];
        int y = (int)d_desiredYs[tid];

        atomicMin(&d_heatmap[y*SIZE + x], 255);
    }
    // printf("-E colorHeatmap\n");
}

__global__ void scaleHeatmap(int* d_heatmap, int* d_scaled_heatmap) {
    /*
        Scale the data for visual representation
    */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int x = 0; x < SIZE; x++) {
        int value = d_heatmap[tid*SIZE + x];
        for (int cellY = 0; cellY < CELLSIZE; cellY++) {
            for (int cellX = 0; cellX < CELLSIZE; cellX++) {
                d_scaled_heatmap[(tid * CELLSIZE + cellY) * SCALED_SIZE + (x * CELLSIZE + cellX)] =
                    value;
            }
        }
    }

    // printf("scaleHeatmap\n");
}

__global__ void filterHeatmap(int* d_scaled_heatmap,
                              int* d_blurred_heatmap) {
    /*
        Apply gaussian blur filter
    */
    // printf("filterHeatmap starts\n");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= 2 && tid < SCALED_SIZE - 2) {
        for (int j = 2; j < SCALED_SIZE - 2; j++) {
            int sum = 0;
            for (int k = -2; k < 3; k++) {
                for (int l = -2; l < 3; l++) {
                    sum += d_w[(2 + k) * SCALED_SIZE + (2 + l)] * d_scaled_heatmap[(tid + k)* SCALED_SIZE + (j + l)];
                }
            }
            int value = sum / WEIGHTSUM;
            d_blurred_heatmap[tid * SCALED_SIZE + j] = 0x00FF0000 | value << 24;
        }
    }
    printf("filterHeatmap\n");
}

void Model::setupHeatmapCUDA() {
    int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	int *shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
	int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));

	heatmap = (int**)malloc(SIZE*sizeof(int*));

	scaled_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
	blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));

	for (int i = 0; i < SIZE; i++) {
		heatmap[i] = hm + SIZE*i;
	}

	for (int i = 0; i < SCALED_SIZE; i++) {
		scaled_heatmap[i] = shm + SCALED_SIZE*i;
		blurred_heatmap[i] = bhm + SCALED_SIZE*i;
	}

    // Allocate and Copy to Device
	cudaMalloc(&d_heatmap, SIZE*SIZE*sizeof(int));
	cudaMemcpy(d_heatmap, heatmap[0], SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&d_scaled_heatmap, SCALED_SIZE*SCALED_SIZE*sizeof(int));
	cudaMemset(d_scaled_heatmap, 0, SCALED_SIZE*SCALED_SIZE*sizeof(int));

	cudaMalloc(&d_blurred_heatmap, SCALED_SIZE*SCALED_SIZE*sizeof(int));
	cudaMemset(d_blurred_heatmap, 0, SCALED_SIZE*SCALED_SIZE*sizeof(int));

	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};

	cudaMemcpyToSymbol(d_w, w, 5 * 5 * sizeof(int));

	// Allocate the desired Xs Ys on device
	cudaMalloc(&d_desiredXs, agents.size()*sizeof(float));
	cudaMalloc(&d_desiredYs, agents.size()*sizeof(float));
    printf("[DEBUG]  setup done!\n");
}

void Model::updateHeatmapCUDA() {

    cudaStream_t stream1;
	cudaStreamCreate(&stream1);


    // heatmap fades
    heatFades<<< SIZE, SIZE, 0, stream1>>>(d_heatmap);
    
  
    // copy desiredXs and desiredYs to device
    cudaMemcpyAsync(d_desiredXs, h_desiredXs,
                    agents.size() * sizeof(float), cudaMemcpyHostToDevice,
                    stream1);
    cudaMemcpyAsync(d_desiredYs, h_desiredYs,
                    agents.size() * sizeof(float), cudaMemcpyHostToDevice,
                    stream1);
    
    
    // heatmap count
    countHeatmap<<<CELLSIZE, SIZE, 0, stream1>>>(d_heatmap, d_desiredXs,
                                                     d_desiredYs, agents.size());

    printf("[DEBUG] countHeatmap done!\n");
    // Color Heatmap
    colorHeatmap<<<1, SIZE, 0, stream1>>>(d_heatmap, d_desiredXs, d_desiredYs, agents.size());

    // Scale Heatmap
    scaleHeatmap<<<1, SIZE, 0, stream1>>>(d_heatmap, d_scaled_heatmap);

    filterHeatmap<<<1, SIZE, 0, stream1>>>(d_scaled_heatmap, d_blurred_heatmap);

    cudaMemcpyAsync(blurred_heatmap[0], d_blurred_heatmap, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyDeviceToHost, stream1);
    
	cudaStreamDestroy(stream1);


   
}
}  // namespace Ped