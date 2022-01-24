//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_waypoint.h"

#include <cmath>

#include <stdlib.h>

// initialize static variables
int Ped::Twaypoint::staticid = 0;

// Constructor: Sets some intial values. The agent has to pass within the given radius.
Ped::Twaypoint::Twaypoint(double px, double py, double pr) : id(staticid++), x(px), y(py), r(pr) {};

// Constructor - sets the most basic parameters.
Ped::Twaypoint::Twaypoint() : id(staticid++), x(0), y(0), r(1) {};

Ped::Twaypoint::~Twaypoint() {};


