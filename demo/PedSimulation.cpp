///////////////////////////////////////////////////
// Low Level Parallel Programming 2016.
//
//     ==== There is no need to change this file ====
// 

#include "PedSimulation.h"
#include <iostream>
#include <QApplication>

#include <stdlib.h>

using namespace std;

PedSimulation::PedSimulation(Ped::Model &model_, MainWindow *window_, bool timing_mode) : model(model_), window(window_), maxSimulationSteps(-1), timingMode(timing_mode)
{
	tickCounter = 0;
}

int PedSimulation::getTickCount() const
{
	return tickCounter;
}
void PedSimulation::simulateOneStep()
{
	tickCounter++;
	model.tick();
    if (!timingMode)
        window->paint();
	if (maxSimulationSteps-- == 0)
	{
        if(timingMode)
            exit(0);
        else
            QApplication::quit();
	}
}

void PedSimulation::runSimulation(int maxNumberOfStepsToSimulate)
{
    if (timingMode)
        runSimulationWithoutQt(maxNumberOfStepsToSimulate);
    else
        runSimulationWithQt(maxNumberOfStepsToSimulate);
}

void PedSimulation::runSimulationWithQt(int maxNumberOfStepsToSimulate)
{
	maxSimulationSteps = maxNumberOfStepsToSimulate;

	//movetimer.setInterval(50); // Limits the simulation to 20 FPS (if one so whiches).
	QObject::connect(&movetimer, SIGNAL(timeout()), this, SLOT(simulateOneStep()));
	movetimer.start();
}

void PedSimulation::runSimulationWithoutQt(int maxNumberOfStepsToSimulate)
{

	maxSimulationSteps = maxNumberOfStepsToSimulate;
	for (int i = 0; i < maxSimulationSteps; i++)
	{
		tickCounter++;
		model.tick();
	}
}

#include "PedSimulation.moc"