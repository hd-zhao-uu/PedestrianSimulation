//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <map>
#include <set>
#include <vector>
#include <atomic>

#include "ped_agent.h"
#include "ped_agent_cuda.h"
#include "ped_agent_soa.h"

namespace Ped {
class Tagent;

// The implementation modes for Assignment 1 + 2:
// chooses which implementation to use for tick()
enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ, REGION };

class Model {
   public:
    // Sets everything up
    void setup(std::vector<Tagent*> agentsInScenario,
               std::vector<Twaypoint*> destinationsInScenario,
               IMPLEMENTATION implementation);

    // FOR ASSIGNMENT 1
    // threadNum setter/getter
    void setThreadNum(const int threadNum) { this->threadNum = threadNum; }

    int getThreadNum() const { return threadNum; }

    // Coordinates a time step in the scenario: move all agents by one step (if
    // applicable).
    void tick();

    // Returns the agents of this scenario
    const std::vector<Tagent*> getAgents() const { return agents; };

    // Adds an agent to the tree structure
    void placeAgent(const Ped::Tagent* a);

    // Cleans up the tree and restructures it. Worth calling every now and then.
    void cleanup();
    ~Model();

    // Returns the heatmap visualizing the density of agents
    int const* const* getHeatmap() const { return blurred_heatmap; };
    int getHeatmapSize() const;

   private:
    // Denotes which implementation (sequential, parallel implementations..)
    // should be used for calculating the desired positions of
    // agents (Assignment 1)
    IMPLEMENTATION implementation;

    int threadNum = 8;

    // The agents in this scenario
    std::vector<Tagent*> agents;

    // The waypoints in this scenario
    std::vector<Twaypoint*> destinations;

    // Moves an agent towards its next position
    void move(Ped::Tagent* agent);

    Ped::TagentSOA* agentSOA = nullptr;
    Ped::TagentCUDA* agentCUDA = nullptr;

    ////////////
    /// Everything below here won't be relevant until Assignment 3
    ///////////////////////////////////////////////
    std::vector<int> agentsIdx;
    void sortAgents();
    void sortAgentsY();

    int offsetX = 50;
    int offsetY = 50;
    
    std::vector<std::vector<int>> stateBoard;

    int& stateUnit(int x, int y) {
        return stateBoard[x + this->offsetX][y + this->offsetY];
    }

    void move(int& rStart, int& rEnd);

    // Returns the set of neighboring agents for the specified position
    set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

    ////////////
    /// Everything below here won't be relevant until Assignment 4
    ///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE* CELLSIZE

    // The heatmap representing the density of agents
    int** heatmap;
    int *d_heatmap;

    // The scaled heatmap that fits to the view
    int** scaled_heatmap;
    int *d_scaled_heatmap;

    // The final heatmap: blurred and scaled to fit the view
    int** blurred_heatmap;
    int *d_blurred_heatmap;

    float *h_desiredXs;
    float *h_desiredYs;
    float *d_desiredXs;
    float *d_desiredYs;
    void setupHeatmapCUDA();
    void updateHeatmapCUDA();
    void freeCUDAMem();

    void setupHeatmapSeq();
    void updateHeatmapSeq();
};
}  // namespace Ped
#endif
