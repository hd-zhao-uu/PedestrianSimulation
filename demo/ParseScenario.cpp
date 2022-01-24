//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017.
//


#include "ParseScenario.h"
#include <string>
#include <iostream>

#include <stdlib.h>

// Comparator used to identify if two agents differ in their position
bool positionComparator(Ped::Tagent *a, Ped::Tagent *b) {
	// True if positions of agents differ
	return (a->getX() < b->getX()) || ((a->getX() == b->getX()) && (a->getY() < b->getY()));
}

/// object constructor
/// \date    2011-01-03
ParseScenario::ParseScenario(QString filename) : QObject(0)
{
	QFile file(filename);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		std::cout << "Warning: file not found or invalid: " << filename.toStdString() << "." << std::endl;
		return;
	}

	while (!file.atEnd())
	{
		QByteArray line = file.readLine();
		processXmlLine(line);
	}

	// Hack! Do not allow agents to be on the same position. Remove duplicates from scenario and free the memory.
	bool(*fn_pt)(Ped::Tagent*, Ped::Tagent*) = positionComparator;
	std::set<Ped::Tagent*, bool(*)(Ped::Tagent*, Ped::Tagent*)> agentsWithUniquePosition(fn_pt);
	int duplicates = 0;
	for (auto agent : agents)
	{
		if (agentsWithUniquePosition.find(agent) == agentsWithUniquePosition.end())
		{
			agentsWithUniquePosition.insert(agent);
		}
		else
		{
			delete agent;
			duplicates += 1;
		}
	}
	if (duplicates > 0)
	{
		std::cout << "Note: removed " << duplicates << " duplicates from scenario." << std::endl;
	}
	agents = std::vector<Ped::Tagent*>(agentsWithUniquePosition.begin(), agentsWithUniquePosition.end());
}

vector<Ped::Tagent*> ParseScenario::getAgents() const
{
	return agents;
}


std::vector<Ped::Twaypoint*> ParseScenario::getWaypoints()
{
	std::vector<Ped::Twaypoint*> v; //
	for (auto p : waypoints)
	{
		v.push_back((p.second));
	}
	return std::move(v);
}

/// Called for each line in the file
void ParseScenario::processXmlLine(QByteArray dataLine)
{
	xmlReader.addData(dataLine);

	while (!xmlReader.atEnd())
	{
		xmlReader.readNext();
		// new definition
		if (xmlReader.isStartElement())
		{
			handleXmlStartElement();
		}
		// closing of definition
		else if (xmlReader.isEndElement())
		{
			handleXmlEndElement();
		}
	}
}

void ParseScenario::handleXmlStartElement()
{
	// New waypoint definition
	if (xmlReader.name() == "waypoint")
	{
		createWaypoint();
	}

	// New agents to add to scenario
	else if (xmlReader.name() == "agent")
	{
		createAgents();
	}

	// Add waypoint that was defined earlier by "createWaypoint"
	// to all agents
	else if (xmlReader.name() == "addwaypoint")
	{
		// Get waypoint by id
		QString id = readString("id");
		addWaypointToCurrentAgents(id);
	}
	else
	{
		// nop, unknown, ignore
	}
}

void ParseScenario::handleXmlEndElement()
{
	// If agents were created in this xml tag,
	// then add the temporary agents to the final
	// collection of agents
	if (xmlReader.name() == "agent") {
		Ped::Tagent *a;
		foreach(a, tempAgents)
		{
			agents.push_back(a);
		}
	}
}

void ParseScenario::createWaypoint()
{
	QString id = readString("id");
	double x = readDouble("x");
	double y = readDouble("y");
	double r = readDouble("r");

	Ped::Twaypoint *w = new Ped::Twaypoint(x, y, r);
	waypoints[id] = w;
}

void ParseScenario::createAgents()
{
	double x = readDouble("x");
	double y = readDouble("y");
	int n = readDouble("n");
	double dx = readDouble("dx");
	double dy = readDouble("dy");

	tempAgents.clear();
	for (int i = 0; i < n; ++i)
	{
		int xPos = x + qrand() / (RAND_MAX / dx) - dx / 2;
		int yPos = y + qrand() / (RAND_MAX / dy) - dy / 2;
		Ped::Tagent *a = new Ped::Tagent(xPos, yPos);
		tempAgents.push_back(a);
	}
}

void ParseScenario::addWaypointToCurrentAgents(QString &id)
{
	Ped::Tagent *a;

	// add the waypoint defined by 'id' to
	// agents created in current xml tag
	foreach(a, tempAgents)
	{
		a->addWaypoint(waypoints[id]);
	}
}

double ParseScenario::readDouble(const QString &tag)
{
	return readString(tag).toDouble();
}

QString ParseScenario::readString(const QString &tag)
{
	return xmlReader.attributes().value(tag).toString();
}

#include "ParseScenario.moc"
