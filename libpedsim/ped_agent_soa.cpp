#include "ped_agent_soa.h"
#include <nmmintrin.h>
#include <cmath>

Ped::TagentSOA::TagentSOA(const std::vector<Ped::Tagent*>& agents) {
    /*
        Convert AOS to SOA
    */
    size = agents.size();

    // the calculation for the next position of 4 agents is computed at the same
    // time by a single thread
    // int soaSize = ceil((double)size / 4) * 4;
    int soaSize = size;

    // allocate aligned memory
    xs = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);
    ys = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);
    desiredXs = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);
    desiredYs = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);

    // destXs = (double*)_mm_malloc(sizeof(double) * soaSize, mAlignment);
    // destYs = (double*)_mm_malloc(sizeof(double) * soaSize, mAlignment);
    // destRs = (double*)_mm_malloc(sizeof(double) * soaSize, mAlignment);

    destXs = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);
    destYs = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);
    destRs = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);

    // waypoints = std::vector<std::deque<Twaypoint*>>(soaSize);

    waypoints = std::vector<std::vector<Twaypoint*>>(soaSize);
    currs = (int*)_mm_malloc(sizeof(int) * soaSize, mAlignment);

    // init values
    for (size_t i = 0; i < size; i++) {
        xs[i] = agents[i]->getX();
        ys[i] = agents[i]->getY();
        desiredXs[i] = agents[i]->getDesiredX();
        desiredYs[i] = agents[i]->getDesiredY();

        // currs[i] = 0;
        auto tWaypoints = agents[i]->getWaypoints();
        //waypoints[i] = std::deque<Twaypoint*>(tWaypoints.begin(), tWaypoints.end());
        waypoints[i] =
            std::vector<Twaypoint*>(tWaypoints.begin(), tWaypoints.end());
        
        destXs[i] = agents[i]->getDestination()->getx();
        destYs[i] = agents[i]->getDestination()->gety();
        destRs[i] = agents[i]->getDestination()->getr();

        // Ped::Twaypoint* nextDest = nullptr;
        // waypoints[i].push_back(dest);
        // nextDest = waypoints[i].front();
        // destXs[i] = nextDest->getx();
        // destYs[i] = nextDest->gety();
        // destRs[i] = nextDest->getr();
        // dest = nextDest;
        // waypoints[i].pop_front();
        
    }
}

void Ped::TagentSOA::getNextDestination() {
    Ped::Twaypoint* nextDestination = NULL;
    bool agentReachedDestination = false;

    for (int i = 0; i < size; i++) {
        double diffX = destXs[i] - xs[i];
        double diffY = destYs[i] - ys[i];
        double length = sqrt(diffX * diffX + diffY * diffY);

        agentReachedDestination = length < destRs[i];

        if (agentReachedDestination) {
            // agent has reached destination (or has no current destination); get next destination if available
            // waypoints[i].push_back(this->dest);
            // Ped::Twaypoint* tDest = waypoints[i].front();
            currs[i] =(currs[i] + 1) % waypoints[i].size();
            Ped::Twaypoint* tDest = waypoints[i][currs[i]];
            destXs[i] = tDest->getx();
            destYs[i] = tDest->gety();
            destRs[i] = tDest->getr();
            // waypoints[i].pop_front();
        }
    }
}

void Ped::TagentSOA::computeNextDesiredPosition() {
    // SIMD
    this->getNextDestination();

    for (size_t i = 0; i < size; i += 4) {
        // __m128d destX, destY;
        // destX = _mm_load_pd(&this->destXs[i]);
        // destY = _mm_load_pd(&this->destYs[i]);

        __m128 destX, destY;
        destX = _mm_load_ps(&this->destXs[i]);
        destY = _mm_load_ps(&this->destYs[i]);

        __m128 x, y;
        x = _mm_load_ps(&this->xs[i]);
        y = _mm_load_ps(&this->ys[i]);

        __m128 diffX, diffY, len;
        diffX = _mm_sub_ps(destX, x);
        diffY = _mm_sub_ps(destY, y);
        len = _mm_sqrt_ps(
            _mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));

        __m128 desiredX, desiredY;
        desiredX = _mm_round_ps(_mm_add_ps(x, _mm_div_ps(diffX, len)),
                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        desiredY = _mm_round_ps(_mm_add_ps(y, _mm_div_ps(diffY, len)),
                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        _mm_store_ps(&this->desiredXs[i], desiredX);
        _mm_store_ps(&this->desiredYs[i], desiredY);
    }
}

void Ped::TagentSOA::computeNextDesiredPositionAndMove() {
    // SIMD
    this->getNextDestination();

    for (size_t i = 0; i < size; i += 4) {
        // __m128d destX, destY;
        // destX = _mm_load_pd(&this->destXs[i]);
        // destY = _mm_load_pd(&this->destYs[i]);

        __m128 destX, destY;
        destX = _mm_load_ps(&this->destXs[i]);
        destY = _mm_load_ps(&this->destYs[i]);

        __m128 x, y;
        x = _mm_load_ps(&this->xs[i]);
        y = _mm_load_ps(&this->ys[i]);

        __m128 diffX, diffY, len;
        diffX = _mm_sub_ps(destX, x);
        diffY = _mm_sub_ps(destY, y);
        len = _mm_sqrt_ps(
            _mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));

        __m128 desiredX, desiredY;
        desiredX = _mm_round_ps(_mm_add_ps(x, _mm_div_ps(diffX, len)),
                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        desiredY = _mm_round_ps(_mm_add_ps(y, _mm_div_ps(diffY, len)),
                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        _mm_store_ps(&this->desiredXs[i], desiredX);
        _mm_store_ps(&this->desiredYs[i], desiredY);

        _mm_store_ps(&this->xs[i], desiredX);
        _mm_store_ps(&this->ys[i], desiredY);
    }
}