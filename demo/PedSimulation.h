///////////////////////////////////////////////////
// Low Level Parallel Programming 2016.
//
//     ==== Don't change this file! ====
//
// PedSimulation wraps the lipbedsim library and
// provides functionality to start the crowd simulation.
//
#ifndef _timer_h_
#define _timer_h_

#include <QObject>
#include <QTimer>
#include "ped_model.h"
#include "MainWindow.h"
// Driver for updating the world
class PedSimulation : public QObject{
	Q_OBJECT

public:
	PedSimulation(Ped::Model &model, MainWindow &window);
	PedSimulation() = delete;
    ~PedSimulation() {}

	// Running simulation without GUI. Use for profiling.
	void runSimulationWithoutQt(int maxNumberOfStepsToSimulate);

	// Running simulation with GUI. Use for visualization.
	void runSimulationWithQt(int maxNumberOfStepsToSimulate);
	int getTickCount() const;
	public slots:
	// Performs one simulation step
	void simulateOneStep();

private:
	Ped::Model &model;
	MainWindow &window;
	QTimer movetimer;
	int maxSimulationSteps;
	int tickCounter;
};
#endif
