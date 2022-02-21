#ifndef _ped_agent_cuda_soa_h_
#define _ped_agent_cuda_soa_h_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <deque>
#include <vector>
#include "ped_agent.h"
#include "ped_waypoint.h"

using namespace std;

namespace Ped {

// Structure of Array
struct TagentCUDA {
    const size_t mAlignment = 16;

    size_t size, soaSize;

    // host
    float *xs, *ys;
    float *desiredXs, *desiredYs;
    float *destXs, *destYs, *destRs;
    int* currs;
    std::vector<std::vector<Ped::Twaypoint> > waypoints;

    // device
    float *xsDevice, *ysDevice;
    float *destXsDevice, *destYsDevice, *destRsDevice;

    TagentCUDA(const std::vector<Ped::Tagent*>& agents);
    ~TagentCUDA();

    void getNextDestination();
    void copyDataToDevice();

    void computeAndMove();
};

}  // namespace Ped

#endif