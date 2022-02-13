#include "ped_agent_cuda.h"

#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


Ped::TagentCUDA::TagentCUDA(const std::vector<Ped::Tagent*>& agents) {
    /*
        Convert AOS to SOA
    */
    this->size = agents.size();

    // the calculation for the next position of 4 agents is computed at the same
    // time by a single thread
    this->soaSize = ceil((double)size / 4) * 4;
    // int soaSize = size;

    xs = (float*) malloc(sizeof(float) * soaSize);
    ys = (float*) malloc(sizeof(float) * soaSize);
    desiredXs =  (float*) malloc(sizeof(float) * soaSize);
    desiredYs = (float*) malloc(sizeof(float) * soaSize);
    destXs = (float*) malloc(sizeof(float) * soaSize);
    destYs = (float*) malloc(sizeof(float) * soaSize);
    destRs = (float*) malloc(sizeof(float) * soaSize);
    currs = (int*) malloc(sizeof(int) * soaSize);
    
    this->waypoints = std::vector<std::vector<Twaypoint*>>(soaSize);
    

    // allocate on device
    cudaMalloc(&xsDevice, sizeof(float) * soaSize);
    cudaMalloc(&ysDevice, sizeof(float) * soaSize);

    cudaMalloc(&destXsDevice, sizeof(float) * soaSize);
    cudaMalloc(&destYsDevice, sizeof(float) * soaSize);
    cudaMalloc(&destRsDevice, sizeof(float) * soaSize);

    // init values
    for (size_t i = 0; i < size; i++) {
        xs[i] = agents[i]->getX();
        ys[i] = agents[i]->getY();
        desiredXs[i] = agents[i]->getDesiredX();
        desiredYs[i] = agents[i]->getDesiredY();

        std::deque<Twaypoint*> tWaypoints = agents[i]->getWaypoints();

        waypoints[i] =
            std::vector<Twaypoint*>(tWaypoints.begin(), tWaypoints.end());
        currs[i] = 0;
    }
}

void Ped::TagentCUDA::getNextDestination() {
    Ped::Twaypoint* nextDestination = NULL;

    for (int i = 0; i < size; i++) {
        bool agentReachedDestination = false;

        double diffX = destXs[i] - xs[i];
        double diffY = destYs[i] - ys[i];
        double length = sqrt(diffX * diffX + diffY * diffY);

        agentReachedDestination = length < destRs[i];

        if (agentReachedDestination && !waypoints[i].empty()) {
            // agent has reached destination (or has no current destination);
            // get next destination if available
            Ped::Twaypoint* dest = waypoints[i][currs[i]];
            destXs[i] = dest->getx();
            destYs[i] = dest->gety();
            destRs[i] = dest->getr();
            currs[i] = (currs[i] + 1) % waypoints[i].size();
        }
    }
}

void Ped::TagentCUDA::copyDataToDevice() {
    size_t bytes = sizeof(float) * soaSize;
    // copy host data to device
    cudaMemcpy(xsDevice, xs,bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ysDevice, ys,bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(destXsDevice, destXs, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(destYsDevice, destYs, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(destRsDevice, destRs, bytes, cudaMemcpyHostToDevice);
}



Ped::TagentCUDA::~TagentCUDA() {

    free(xs);
    free(ys);
    free(desiredXs);
    free(desiredYs);
    free(destXs);
    free(destYs);
    free(destRs);
    free(currs);

    cudaFree(xsDevice);
    cudaFree(ysDevice);
    cudaFree(destXsDevice);
    cudaFree(destYsDevice);
    cudaFree(destRsDevice);
}

