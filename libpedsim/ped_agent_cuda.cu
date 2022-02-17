#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ped_agent_cuda.h"

__global__ void computeAndMoveCUDA(float* xsDevice,
                                   float* ysDevice,
                                   float* destXsDevice,
                                   float* destYsDevice,
                                   float* destRsDevice,
                                   int size) {
    int idx = blockIdx.x * blockDim.y * threadIdx.x;
    if (idx < size) {
        float diffX = destXsDevice[idx] - xsDevice[idx];
        float diffY = destYsDevice[idx] - ysDevice[idx];
        float len = sqrtf(diffX * diffX + diffY * diffY);
        xsDevice[idx] = (int)roundf(xsDevice[idx] + diffX / len);
        ysDevice[idx] = (int)roundf(ysDevice[idx] + diffY / len);
    }
}

void Ped::TagentCUDA::computeAndMove() {
    this->getNextDestination();

    // copy data from host to device
    this->copyDataToDevice();

    int blocks = (int)ceilf((float)this->size / 1024);
    // run kernel
    computeAndMoveCUDA<<<blocks, 1024>>>(this->xsDevice, this->ysDevice,
                                         this->destXsDevice, this->destYsDevice,
                                         this->destRsDevice, this->size);
    // copy results from device to host
    size_t bytes = sizeof(float) * soaSize;
    cudaMemcpy(xs, xsDevice, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ys, ysDevice, bytes, cudaMemcpyHostToDevice);
}
