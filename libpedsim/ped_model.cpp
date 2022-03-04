//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"

#include <nmmintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stack>
#include <thread>

#include "cuda_testkernel.h"
#include "ped_agent_cuda.h"
#include "ped_agent_soa.h"
#include "ped_model.h"
#include "ped_waypoint.h"

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario,
                       std::vector<Twaypoint*> destinationsInScenario,
                       IMPLEMENTATION implementation) {
    // Convenience test: does CUDA work on this machine?
    cuda_test();

    // Set
    agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(),
                                       agentsInScenario.end());

    // Set up destinations
    destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(),
                                                destinationsInScenario.end());

    // Sets the chosen implemenation. Standard in the given code is SEQ
    this->implementation = implementation;

    // Set up heatmap (relevant for Assignment 4)
    setupHeatmapSeq();
}

void Ped::Model::tick() {
    // EDIT HERE FOR ASSIGNMENT 1
    auto a1Move = [](Ped::Tagent* agent) {
        // retrieve the agent and calculate its next desired position
        agent->computeNextDesiredPosition();

        int dX = agent->getDesiredX(), dY = agent->getDesiredY();
        // set its position to the calculated desired one
        agent->setX(dX);
        agent->setY(dY);
    };

    // lambda function for threads
    auto pFunc = [&](int tId, int start, int end,
                     std::vector<Ped::Tagent*>& agents) {
        for (int i = start; i <= end; i++) {
            // retrieve the agent and calculate its next desired position
            Ped::Tagent* agent = agents[i];
            agent->computeNextDesiredPosition();

            int dX = agent->getDesiredX(), dY = agent->getDesiredY();
            // set its position to the calculated desired one
            agent->setX(dX);
            agent->setY(dY);
        }
    };

    int agentSize = agents.size();
    int threadNum = getThreadNum();
    switch (implementation) {
        case SEQ: {
            for (int i = 0; i < agentSize; i++) {
                // a1Move(agents[i]);
                agents[i]->computeNextDesiredPosition();
                move(agents[i]);
            }
            updateHeatmapSeq();
        } break;

        case PTHREAD: {
            int agentsPerThread = agentSize / threadNum;
            int agentLeft = agentSize % threadNum;

            std::thread* threads = new std::thread[threadNum];
            int start, end;
            // allocation strategy:
            // 	if `agentSize` cannot be evenly divided by `threadNum`, then
            // 	the first `agentLeft` threads will handle 1 more agent than
            // other the threads.
            for (int i = 0; i < threadNum; i++) {
                if (i < agentLeft) {
                    start = i * agentsPerThread + i;
                    end = start + agentsPerThread;
                } else {
                    start = i * agentsPerThread + agentLeft;
                    end = start + agentsPerThread - 1;
                }

                threads[i] =
                    std::thread(pFunc, i, start, end, std::ref(agents));
            }

            for (int i = 0; i < threadNum; i++) {
                threads[i].join();
            }

            delete[] threads;

        } break;

        case OMP: {
            int i;
#pragma omp parallel for shared(agents) num_threads(threadNum) schedule(static)
            for (i = 0; i < agentSize; i++) {
                a1Move(agents[i]);
            }

        } break;

        case VECTOR: {
            if (!agentSOA) {
                for (int i = 0; i < agents.size(); i++) {
                    agents[i]->computeNextDesiredPosition();

                    int dX = agents[i]->getDesiredX(),
                        dY = agents[i]->getDesiredY();
                    // set its position to the calculated desired one
                    agents[i]->setX(dX);
                    agents[i]->setY(dY);
                }
                agentSOA = new Ped::TagentSOA(agents);
            }
            agentSOA->setThreads(threadNum);
            agentSOA->computeAndMove();
            float *xs = agentSOA->xs, *ys = agentSOA->ys;

// For Painting
#pragma omp parallel for shared(agents) num_threads(threadNum) schedule(static)
            for (size_t i = 0; i < agents.size(); i++) {
                agents[i]->setX(xs[i]);
                agents[i]->setY(ys[i]);
            }

        } break;

        case CUDA: {
            if (!agentCUDA) {
                for (int i = 0; i < agents.size(); i++) {
                    agents[i]->computeNextDesiredPosition();

                    int dX = agents[i]->getDesiredX(),
                        dY = agents[i]->getDesiredY();
                    // set its position to the calculated desired one
                    agents[i]->setX(dX);
                    agents[i]->setY(dY);
                }
                agentCUDA = new Ped::TagentCUDA(agents);
            }

            agentCUDA->computeAndMove();
            float *xs = agentCUDA->xs, *ys = agentCUDA->ys;

#pragma omp parallel for shared(agents) num_threads(threadNum) schedule(static)
            for (size_t i = 0; i < agents.size(); i++) {
                agents[i]->setX(xs[i]);
                agents[i]->setY(ys[i]);
            }

        } break;

        case REGION: {
            if (!agentSOA) {
                for (int i = 0; i < agents.size(); i++) {
                    agents[i]->computeNextDesiredPosition();

                    int dX = agents[i]->getDesiredX(),
                        dY = agents[i]->getDesiredY();
                    // set its position to the calculated desired one
                    agents[i]->setX(dX);
                    agents[i]->setY(dY);
                }
                agentSOA = new Ped::TagentSOA(agents);
                agentSOA->setThreads(threadNum);
                sortAgents();

                // Ped::Tagent* minAgent = agents[agentsIdx[0]];
                Ped::Tagent* maxAgent = agents[agentsIdx[agents.size() - 1]];

                int boardX = ceil((double)maxAgent->getX() / 100 + 3) * 100;
                int boardY = ceil((double)maxAgent->getY() / 100 + 3) * 100;

                // printf("boardX= %d, boardY = %d\n", boardX, boardY);
                // printf("%d %d\n", maxAgent->getX(), maxAgent->getY());

                // state board: -1 => no agent occupies, otherwise it records
                // the occupier
                stateBoard = std::vector<std::vector<int>>(
                    boardX, std::vector<int>(boardY, -1));

                for (int i = 0; i < agents.size(); i++) {
                    int sx = agents[i]->getX(), sy = agents[i]->getY();
                    // stateBoard[sx][sy] = i;
                    stateUnit(sx, sy) = i;
                }
            }

            sortAgents();
            agentSOA->computeNextDesiredPosition();

            omp_set_num_threads(threadNum);
#pragma omp parallel
            {
                int agentsInRegion = ceil((double)agents.size() / threadNum);
                int threadId = omp_get_thread_num();
                int rStart = threadId * agentsInRegion;
                int rEnd = rStart + agentsInRegion < agents.size()
                               ? rStart + agentsInRegion
                               : agents.size();
                // printf("agentsSize = %d, agentsInRegion= %d, thread = %d,
                // start = %d, end = %d\n", agents.size(), agentsInRegion,
                // threadId, rStart, rEnd);
                move(rStart, rEnd);
            }

            float *xs = agentSOA->xs, *ys = agentSOA->ys;
#pragma omp parallel for shared(agents) num_threads(threadNum) schedule(static)
            for (size_t i = 0; i < agents.size(); i++) {
                agents[i]->setX(xs[i]);
                agents[i]->setY(ys[i]);
            }

        } break;

        default:
            break;
    }
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

void Ped::Model::sortAgents() {
    agentsIdx = vector<int>(agents.size());
    std::iota(agentsIdx.begin(), agentsIdx.end(), 0);

    // get the sorted index
    sort(agentsIdx.begin(), agentsIdx.end(),
         [=](const int& i, const int& j) -> bool {
             // if (agents[i]->getX() != agents[j]->getX()) return
             // agents[i]->getX() < agents[j]->getX(); return agents[i]->getY()
             // < agents[i]->getY();
             return agents[i]->getX() < agents[j]->getX();
         });
}

void Ped::Model::move(int& rStart, int& rEnd) {
    float rangeXStart = agentSOA->xs[agentsIdx[rStart]];
    if (rEnd == agentsIdx.size())
        rEnd = rEnd - 1;
    float rangeXEnd = agentSOA->xs[agentsIdx[rEnd]];

    // float rangeYStart = agentSOA->ys[agentsIdx[rStart]];
    // float rangeYEnd = agentSOA->ys[agentsIdx[rEnd]];

    std::srand(std::time(0));
    random_shuffle(agentsIdx.begin() + rStart, agentsIdx.begin() + rEnd);

    for (int i = rStart; i < rEnd; i++) {
        int aId = agentsIdx[i];
        int x = agentSOA->xs[aId];
        int y = agentSOA->ys[aId];
        int desiredX = agentSOA->desiredXs[aId];
        int desiredY = agentSOA->desiredYs[aId];

        // Compute the three alternative positions that would bring the agent
        // closer to his desiredPosition, starting with the desiredPosition
        // itself
        std::pair<int, int> p0, p1, p2;
        p0 = std::make_pair(desiredX, desiredY);

        auto diffX = desiredX - x;
        auto diffY = desiredY - y;

        if (diffX == 0 || diffY == 0) {
            // Agent wants to walk straight to North, South, West or East
            p1 = std::make_pair(desiredX + diffY, desiredY + diffX);
            p2 = std::make_pair(desiredX - diffY, desiredY - diffX);
        } else {
            // Agent wants to walk diagonally
            p1 = std::make_pair(desiredX, y);
            p2 = std::make_pair(x, desiredY);
        }
        auto pCandidates = std::vector<std::pair<int, int>>({p0, p1, p2});

        // Find the first empty alternative position
        for (auto position : pCandidates) {
            int px, py;
            std::tie(px, py) = position;

            bool isInRegion = px > rangeXStart && px < rangeXEnd;
            // && py > rangeYStart && py < rangeYEnd;
            if (isInRegion) {
                // agents can move freely in the region
                if (stateUnit(px, py) == -1) {
                    stateUnit(px, py) = aId;
                    stateUnit(x, y) = -1;
                    agentSOA->xs[aId] = px;
                    agentSOA->ys[aId] = py;
                    break;
                }
            } else {
                if (__sync_bool_compare_and_swap(&stateUnit(px, py), -1, aId)) {
                    stateUnit(x, y) = -1;
                    agentSOA->xs[aId] = px;
                    agentSOA->ys[aId] = py;
                    break;
                }
            }
        }
    }
}

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent* agent) {
    // Search for neighboring agents
    set<const Ped::Tagent*> neighbors =
        getNeighbors(agent->getX(), agent->getY(), 2);

    // Retrieve their positions
    std::vector<std::pair<int, int>> takenPositions;
    for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin();
         neighborIt != neighbors.end(); ++neighborIt) {
        std::pair<int, int> position((*neighborIt)->getX(),
                                     (*neighborIt)->getY());
        takenPositions.push_back(position);
    }

    // Compute the three alternative positions that would bring the agent
    // closer to his desiredPosition, starting with the desiredPosition itself
    std::vector<std::pair<int, int>> prioritizedAlternatives;
    std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
    prioritizedAlternatives.push_back(pDesired);

    int diffX = pDesired.first - agent->getX();
    int diffY = pDesired.second - agent->getY();
    std::pair<int, int> p1, p2;
    if (diffX == 0 || diffY == 0) {
        // Agent wants to walk straight to North, South, West or East
        p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
        p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
    } else {
        // Agent wants to walk diagonally
        p1 = std::make_pair(pDesired.first, agent->getY());
        p2 = std::make_pair(agent->getX(), pDesired.second);
    }
    prioritizedAlternatives.push_back(p1);
    prioritizedAlternatives.push_back(p2);

    // Find the first empty alternative position
    for (std::vector<pair<int, int>>::iterator it =
             prioritizedAlternatives.begin();
         it != prioritizedAlternatives.end(); ++it) {
        // If the current position is not yet taken by any neighbor
        if (std::find(takenPositions.begin(), takenPositions.end(), *it) ==
            takenPositions.end()) {
            // Set the agent's position
            agent->setX((*it).first);
            agent->setY((*it).second);

            break;
        }
    }
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents
/// (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {
    // create the output list
    // ( It would be better to include only the agents close by, but this
    // programmer is lazy.)
    return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
    // Nothing to do here right now.
}

Ped::Model::~Model() {
    std::for_each(agents.begin(), agents.end(),
                  [](Ped::Tagent* agent) { delete agent; });
    std::for_each(destinations.begin(), destinations.end(),
                  [](Ped::Twaypoint* destination) { delete destination; });
    if (agentSOA != nullptr)
        delete agentSOA;
    if (agentCUDA != nullptr)
        delete agentCUDA;
}
