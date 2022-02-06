#ifndef _ped_agent_soa_h_
#define _ped_agent_soa_h_

#include <deque>
#include <vector>
#include "ped_agent.h"
#include "ped_waypoint.h"

using namespace std;

namespace Ped {

// Structure of Array
struct TagentSOA {
    const size_t mAlignment = 16;

    size_t size;

    float *xs, *ys;
    float *desiredXs, *desiredYs;

    // split Twaypoint
    // double *destXs, *destYs, *destRs;
    float *destXs, *destYs, *destRs;

    Ped::Twaypoint* dest;

    // std::deque<Twaypoint*>* waypoints;
    // std::vector<std::deque<Ped::Twaypoint*>> waypoints;  

    int* currs;
    std::vector<std::vector<Ped::Twaypoint*>> waypoints;  


    TagentSOA(const std::vector<Ped::Tagent*>& agents);

    void getNextDestination();

    void computeNextDesiredPosition();

    void computeNextDesiredPositionAndMove();
};

}  // namespace Ped

#endif