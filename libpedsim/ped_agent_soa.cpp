#include "ped_agent_soa.h"
#include <nmmintrin.h>
#include <omp.h>
#include <cmath>
#include <cstdio>

Ped::TagentSOA::TagentSOA(const std::vector<Ped::Tagent*>& agents) {
    /*
        Convert AOS to SOA
    */
    
    this->size = agents.size();

    // the calculation for the next position of 4 agents is computed at the same
    // time by a single thread
    int soaSize = ceil((double)size / 4) * 4;

    // allocate aligned memory
    xs = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);
    ys = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);
    desiredXs = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);
    desiredYs = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);

    destXs = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);
    destYs = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);
    destRs = (float*)_mm_malloc(sizeof(float) * soaSize, mAlignment);
    currs = (int*)_mm_malloc(sizeof(int) * soaSize, mAlignment);

    waypoints = std::vector<std::vector<Twaypoint> >(soaSize);

// init values
    for (size_t i = 0; i < size; i++) {
        xs[i] = agents[i]->getX();
        ys[i] = agents[i]->getY();
        desiredXs[i] = agents[i]->getDesiredX();
        desiredYs[i] = agents[i]->getDesiredY();
        destXs[i] = agents[i]->getDestination()->getx();
        destYs[i] = agents[i]->getDestination()->gety();
        destRs[i] = agents[i]->getDestination()->getr();

        currs[i] = agents[i]->getCurr();
        std::vector<Ped::Twaypoint> tWaypoints = agents[i]->getWaypoints();
        waypoints[i] = std::vector<Ped::Twaypoint>(tWaypoints.begin(), tWaypoints.end());

    }
}

void Ped::TagentSOA::setThreads(const int threadNum) {
    this->threadNum = threadNum;
}

void Ped::TagentSOA::getNextDestination() {
    Ped::Twaypoint* nextDestination = NULL;

#pragma omp parallel for shared(destXs, destYs, xs, ys, currs, destRs, \
                                waypoints) num_threads(threadNum)      \
    schedule(static)
    for (int i = 0; i < size; i++) {
        bool agentReachedDestination = false;

        double diffX = destXs[i] - xs[i];
        double diffY = destYs[i] - ys[i];
        double length = sqrt(diffX * diffX + diffY * diffY);

        agentReachedDestination = length < destRs[i];

        if (agentReachedDestination && !waypoints[i].empty()) {
            // agent has reached destination (or has no current destination);
            // get next destination if available
            int s = waypoints[i].size();
            currs[i] = (currs[i] + 1) % waypoints[i].size();
            Ped::Twaypoint dest = waypoints[i][currs[i]];
            destXs[i] = dest.getx();
            destYs[i] = dest.gety();
            destRs[i] = dest.getr();
            
            
        }
    }
}

void Ped::TagentSOA::computeNextDesiredPosition() {
    // SIMD
    this->getNextDestination();

#pragma omp parallel for shared(destXs, destYs, xs, ys, desiredXs, desiredYs) \
    num_threads(threadNum) schedule(static)
    for (size_t i = 0; i < size; i += 4) {
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
        desiredX = _mm_add_ps(x, _mm_div_ps(diffX, len));
        desiredY = _mm_add_ps(y, _mm_div_ps(diffY, len));
        desiredX = _mm_round_ps(desiredX,
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        desiredY = _mm_round_ps(desiredY,
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        _mm_store_ps(&this->desiredXs[i], desiredX);
        _mm_store_ps(&this->desiredYs[i], desiredY);

        _mm_store_ps(&this->xs[i], desiredX);
        _mm_store_ps(&this->ys[i], desiredY);
    }
}

void Ped::TagentSOA::computeAndMove() {
    // SIMD
    this->getNextDestination();

#pragma omp parallel for shared(destXs, destYs, xs, ys, desiredXs, desiredYs) \
    num_threads(threadNum) schedule(static)
    for (size_t i = 0; i < size; i += 4) {
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
        desiredX = _mm_add_ps(x, _mm_div_ps(diffX, len));
        desiredY = _mm_add_ps(y, _mm_div_ps(diffY, len));
        desiredX = _mm_round_ps(desiredX,
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        desiredY = _mm_round_ps(desiredY,
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        _mm_store_ps(&this->desiredXs[i], desiredX);
        _mm_store_ps(&this->desiredYs[i], desiredY);

        _mm_store_ps(&this->xs[i], desiredX);
        _mm_store_ps(&this->ys[i], desiredY);
    }
}

Ped::TagentSOA::~TagentSOA() {
    _mm_free(xs);
    _mm_free(ys);

    _mm_free(desiredXs);
    _mm_free(desiredYs);

    _mm_free(destXs);
    _mm_free(destYs);
    _mm_free(destRs);

    _mm_free(currs);
}