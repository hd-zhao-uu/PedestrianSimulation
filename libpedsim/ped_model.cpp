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
#include <iostream>
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

    agentSOA = new Ped::TagentSOA(agents);
    agentCUDA = new Ped::TagentCUDA(agents);

    // Set up heatmap (relevant for Assignment 4)
    setupHeatmapSeq();

    for (int i = 0; i < agents.size(); i++)
    {
        if (agents[i]->getX() < 80)
	    {
	        if (agents[i]->getY() < 60)
	        {
	            this->agent1.push_back(agents[i]);
	        }
	        else
	        {
	            this->agent2.push_back(agents[i]);
	        }
	    }
        else
	    {
	        if (agents[i]->getY() < 60)
	        {
	            this->agent3.push_back(agents[i]);
	        }
	        else
	        {
	            this->agent4.push_back(agents[i]);
	        }
	    }
    }

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
            std::vector<Tagent *> emptyAgent;
	        for (int i = 0; i < agents.size(); i++)
	        {
	            agents[i]->computeNextDesiredPosition();
	            move(agents[i], agents, emptyAgent);
	        }
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
            agentSOA->setThreads(threadNum);

            agentSOA->computeNextDesiredPositionAndMove();
            float *xs = agentSOA->xs, *ys = agentSOA->ys;

// For Painting
#pragma omp parallel for shared(agents) num_threads(threadNum) schedule(static)
            for (size_t i = 0; i < agents.size(); i++) {
                agents[i]->setX(xs[i]);
                agents[i]->setY(ys[i]);
            }

        } break;

        case CUDA: {
            agentCUDA->computeAndMove();
            float *xs = agentCUDA->xs, *ys = agentCUDA->ys;
#pragma omp parallel for shared(agents) num_threads(threadNum) schedule(static)
            for (size_t i = 0; i < agents.size(); i++) {
                agents[i]->setX(xs[i]);
                agents[i]->setY(ys[i]);
            }

        } break;

        case TASK: {
            omp_set_num_threads(threadNum);
#pragma omp parallel
{
#pragma omp single nowait
{
#pragma omp task
{
	    for (int i = 0; i < this->agent1.size(); i++){
		    this->agent1[i]->computeNextDesiredPosition();
		    if(checkPosition(agent1[i])){
		      this->border1.push_back(this->agent1[i]);
		      this->agent1.erase(this->agent1.begin() + i);
		      i--;
		    }
		    else {
		      move(this->agent1[i], this->agent1, this->border1);
		    }
		}
}
#pragma omp task
{
	    for (int i = 0; i < this->agent2.size(); i++){
		    this->agent2[i]->computeNextDesiredPosition();
		    if(checkPosition(agent2[i])){
		      this->border2.push_back(this->agent2[i]);
		      this->agent2.erase(this->agent2.begin() + i);
		      i--;
		    }
		    else {
		      move(this->agent2[i], this->agent2, this->border2);
		    }
		}
}
#pragma omp task
{
	    for (int i = 0; i < this->agent3.size(); i++){
		    this->agent3[i]->computeNextDesiredPosition();
		    if(checkPosition(agent3[i])){
		      this->border3.push_back(this->agent3[i]);
		      this->agent3.erase(this->agent3.begin() + i);
		      i--;
		    }
		    else {
		      move(this->agent3[i], this->agent3, this->border3);
		    }
		}
}
#pragma omp task
{
	    for (int i = 0; i < this->agent4.size(); i++){
		    this->agent4[i]->computeNextDesiredPosition();
		    if(checkPosition(agent4[i])){
		      this->border4.push_back(this->agent4[i]);
		      this->agent4.erase(this->agent4.begin() + i);
		      i--;
		    }
		    else {
		      move(this->agent4[i], this->agent4, this->border4);
		    }
		}
}



}
#pragma omp taskwait

}       

    int largestArray = (int)std::max(std::max(this->border1.size(),this->border2.size()),std::max(this->border3.size(),this->border4.size()));

	std::vector<Ped::Tagent *> allBorderPoints;
	allBorderPoints.insert(allBorderPoints.end(), this->border1.begin(), this->border1.end());
	allBorderPoints.insert(allBorderPoints.end(), this->border2.begin(), this->border2.end());
	allBorderPoints.insert(allBorderPoints.end(), this->border3.begin(), this->border3.end());
	allBorderPoints.insert(allBorderPoints.end(), this->border4.begin(), this->border4.end());

	for(int i=0; i < largestArray; i++){
        if(i < this->border1.size()){
            move(this->border1[i], this->agent1, allBorderPoints);
            moveAgentList(this->border1[i]);
	    }
        if(i < this->border2.size()){
            move(this->border2[i], this->agent2, allBorderPoints);
            moveAgentList(this->border2[i]);
	    }
        if(i < this->border3.size()){
            move(this->border3[i], this->agent3, allBorderPoints);
            moveAgentList(this->border3[i]);
	    }
        if(i < this->border4.size()){
            move(this->border4[i], this->agent4, allBorderPoints);
            moveAgentList(this->border4[i]);
	    }   
	}
    this->border1.clear();
	this->border2.clear();
	this->border3.clear();
	this->border4.clear();
    
	break;
}
    default:
            break;
    }
}

void Ped::Model::moveAgentList(Ped::Tagent *agent)
{
    if (agent->getX() < 80){
        if (agent->getY() < 60){
	        agent1.push_back(std::move(agent));
	}
        else{
	        agent2.push_back(std::move(agent));
	    }
    }
    else if (agent->getX() >= 80){
        if (agent->getY() < 60){
	        agent3.push_back(std::move(agent));
	    }
        else{
	        agent4.push_back(std::move(agent));
	    }
    }
}

bool Ped::Model::checkPosition(Ped::Tagent *agent)
{
  if((agent->getDesiredX() >= 78 and
      agent->getDesiredX() < 82) or
     (agent->getDesiredY() >= 58 and
      agent->getDesiredY() < 62))
    {
      return true;
    }
  else
    {
      return false;
    }
}
////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent* agent, std::vector<Ped::Tagent *> agentVector, std::vector<Ped::Tagent *> borderVector) {
    // Search for neighboring agents
    set<const Ped::Tagent*> neighbors =
        getNeighbors(agent->getX(), agent->getY(), 2, agentVector, borderVector);

    // Retrieve their positions
    std::vector<std::pair<int, int> > takenPositions;
    for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin();
         neighborIt != neighbors.end(); ++neighborIt) {
        std::pair<int, int> position((*neighborIt)->getX(),
                                     (*neighborIt)->getY());
        takenPositions.push_back(position);
    }

    // Compute the three alternative positions that would bring the agent
    // closer to his desiredPosition, starting with the desiredPosition itself
    std::vector<std::pair<int, int> > prioritizedAlternatives;
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
    for (std::vector<pair<int, int> >::iterator it =
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
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist, std::vector<Ped::Tagent *> agentVector, std::vector<Ped::Tagent *> borderVector) const{
    // create the output list
    // ( It would be better to include only the agents close by, but this
    // programmer is lazy.)
    std::vector<Ped::Tagent *> allNeighbors(0);

    for (int i = 0; i < agentVector.size(); i++){
        int aX = agentVector[i]->getX();
        int aY = agentVector[i]->getY();

        if (aX < (x + dist) &&
	    aX > (x - dist) &&
	    aY < (y + dist) &&
	    aY > (y - dist) &&
	    (aX != x or aY != y)){
	        allNeighbors.push_back(agentVector[i]);
	    }
    }

    for (int i = 0; i < borderVector.size(); i++){
        int aX = borderVector[i]->getX();
        int aY = borderVector[i]->getY();

        if (aX < (x + dist) &&
	    aX > (x - dist) &&
	    aY < (y + dist) &&
	    aY > (y - dist) &&
	    (aX != x or aY != y)){
	        allNeighbors.push_back(borderVector[i]);
	    }
    }
    return set<const Ped::Tagent*>(allNeighbors.begin(), allNeighbors.end());
}

void Ped::Model::cleanup() {
    // Nothing to do here right now.
}

Ped::Model::~Model() {
    std::for_each(agents.begin(), agents.end(),
                  [](Ped::Tagent* agent) { delete agent; });
    std::for_each(destinations.begin(), destinations.end(),
                  [](Ped::Twaypoint* destination) { delete destination; });
    delete agentSOA;
    delete agentCUDA;
}
