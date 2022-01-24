///////////////////////////////////////////////////
// Low Level Parallel Programming 2016.
//
//     ==== Don't change this file! ====
// 
#ifndef _view_agent_h
#define _view_agent_h

#include <QGraphicsScene>
#include <QGraphicsRectItem>
#include "ped_agent.h"

class ViewAgent{
public:
	ViewAgent(Ped::Tagent * agent, QGraphicsScene * scene);
	void paint(QColor color);
	const std::pair<int, int> getPosition();

private:
	const Ped::Tagent *agent;

	// The rectangle on the GUI representing this agent
	QGraphicsRectItem * rect;
	QGraphicsPixmapItem * bgt_icon;



};

#endif

