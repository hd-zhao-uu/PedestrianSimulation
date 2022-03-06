#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include "ped_model.h"


constexpr int WEIGHTSUM = 273;

__constant__ int d_w[5 * 5];

namespace Ped {

__global__ void heatFades(int* d_heatmap) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
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
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // boundary check
    if (tid < agentSize) {
        int x = (int)d_desiredXs[tid];
        int y = (int)d_desiredYs[tid];
        if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
            return;
        atomicAdd(&d_heatmap[y*SIZE + x], 40);
    }
}

__global__ void colorHeatmap(int* d_heatmap,
                             float* d_desiredXs,
                             float* d_desiredYs, 
                             const int agentSize) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < agentSize) {
        int x = (int)d_desiredXs[tid];
        int y = (int)d_desiredYs[tid];

        atomicMin(&d_heatmap[y*SIZE + x], 255);
    }
}

__global__ void scaleHeatmap(int* d_heatmap, int* d_scaled_heatmap) {
    /*
        Scale the data for visual representation
        Parallize: each thread scale one heatmap pixel
    */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int value = d_heatmap[tid];
    int y = tid / SIZE;
    int x = tid % SIZE;
    for (int cellY = 0; cellY < CELLSIZE; cellY++) {
		for (int cellX = 0; cellX < CELLSIZE; cellX++){
            int s_y = y * CELLSIZE + cellY;
            int s_x = x * CELLSIZE + cellX;
			d_scaled_heatmap[s_y*SCALED_SIZE  + s_x] = value;
		}
	}

}

__global__ void filterHeatmap(int* d_scaled_heatmap,
                              int* d_blurred_heatmap) {
    /*
        Apply gaussian blur filter
        Parallize: parallelize the outer 2 for-loops
    */
    __shared__ int shared_shm [32][32];

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    shared_shm[threadIdx.y][threadIdx.x] = d_scaled_heatmap[y * SCALED_SIZE + x];
    __syncthreads();

    if(2 <= x && x < SCALED_SIZE - 2 && 2 <= y && y < SCALED_SIZE - 2) {
        int sum = 0 ;
        for (int k = -2; k < 3; k++) {
            for (int l = -2; l < 3; l++) {
                int shm_y = threadIdx.y + k;
                int shm_x = threadIdx.x + l;
                int v;
                if(0 <= shm_y && shm_y < 32 && 0 <= shm_x && shm_x < 32)
                    v = shared_shm[shm_y][shm_x];
                else
                    v = d_scaled_heatmap[(y + k) * SCALED_SIZE + x + l];
                sum += d_w[(2 + k) * 5 + (2 + l)] * v;
            }
        }
    int val = sum / 273;
    d_blurred_heatmap[y * SCALED_SIZE + x] = 0x00FF0000 | val << 24;
  }

    
}


__global__ void __filterHeatmap(int* d_scaled_heatmap,
                              int* d_blurred_heatmap) {
    /*
        Apply gaussian blur filter
    */
    // printf("filterHeatmap starts\n");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= 2 && tid < SCALED_SIZE - 2) {
        for (int j = 2; j < SCALED_SIZE - 2; j++) {
            int sum = 0 ;
            for (int k = -2; k < 3; k++) {
                for (int l = -2; l < 3; l++) {
                    sum += d_w[(2 + k) * 5 + (2 + l)] * d_scaled_heatmap[(tid + k)* SCALED_SIZE + (j + l)];
                }
            }
            int value = sum / WEIGHTSUM;
            // printf("%d ", value);
            d_blurred_heatmap[tid * SCALED_SIZE + j] = 0x00FF0000 | value << 24;
        }
    }
    // printf("filterHeatmap done!\n");
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
    printf("[DEBUG] CUDA heatmap setup!\n");
}

void Model::updateHeatmapCUDA() {

    cudaStream_t stream1;
	cudaStreamCreate(&stream1);
    
    // heatmap fades
    dim3 hm_bSize = SIZE;
    dim3 hm_blocks = SIZE;
    heatFades<<<hm_blocks, hm_blocks, 0, stream1>>>(d_heatmap);
    
  
    // copy desiredXs and desiredYs to device
    cudaMemcpyAsync(d_desiredXs, h_desiredXs,
                    agents.size() * sizeof(float), cudaMemcpyHostToDevice,
                    stream1);
    cudaMemcpyAsync(d_desiredYs, h_desiredYs,
                    agents.size() * sizeof(float), cudaMemcpyHostToDevice,
                    stream1);
    
    
    // heatmap count
    countHeatmap<<<hm_blocks, hm_bSize, 0, stream1>>>(d_heatmap, d_desiredXs,
                                                     d_desiredYs, agents.size());

    // Color Heatmap
    colorHeatmap<<<hm_blocks, hm_bSize, 0, stream1>>>(d_heatmap, d_desiredXs, d_desiredYs, agents.size());

    // Scale Heatmap
    scaleHeatmap<<<hm_blocks, hm_bSize, 0, stream1>>>(d_heatmap, d_scaled_heatmap);

    // Filter Heatmap
    dim3 filter_bSize(32, 32);
    dim3 filter_blocks(SCALED_SIZE / filter_bSize.x, SCALED_SIZE / filter_bSize.y);

    filterHeatmap<<<filter_blocks, filter_bSize, 0, stream1>>>(d_scaled_heatmap, d_blurred_heatmap);
    // __filterHeatmap<<<1, SIZE, 0, stream1>>>(d_scaled_heatmap, d_blurred_heatmap);

    cudaMemcpyAsync(blurred_heatmap[0], d_blurred_heatmap, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyDeviceToHost, stream1);
    
	cudaStreamDestroy(stream1);


   
}
}  // namespace Ped