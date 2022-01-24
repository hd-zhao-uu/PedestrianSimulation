SOURCES=$(shell echo *.cpp)
OBJECTS=$(SOURCES:.cpp=.o)

LIBPEDSOURCES=$(shell echo ../libpedsim/*.cpp)
LIBPEDOBJECTS=$(LIBPEDSOURCES:.cpp=.o)

LIBPEDCUDASOURCES=$(shell echo ../libpedsim/*.cu)
LIBPEDCUDAOBJECTS=$(LIBPEDSOURCES:.cu=.co)

TARGET=demo
QT_HEADERS=$(shell qmake -query QT_INSTALL_HEADERS)
QTINCLUDES=-I$(QT_HEADERS) -I$(QT_HEADERS)/QtWidgets -I$(QT_HEADERS)/QtGui -I$(QT_HEADERS)/QtCore
INCPATH=-I../libpedsim
LIBPATH=-L../libpedsim
CXXFLAGS=-fPIC $(INCPATH) $(QTINCLUDES) $(LIBPATH) -fopenmp
LIBS = -lQt5Widgets -lQt5Gui -lQt5Core -lpedsim -lcudart
LDFLAGS+="-Wl,-rpath,$(PWD)/libpedsim,-rpath,$(PWD)/../libpedsim"

MOCFILES=ParseScenario.moc PedSimulation.moc


all: $(TARGET)

$(TARGET): $(OBJECTS) $(MOCFILES)
	g++ $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LIBS) $(LDFLAGS)

%.o: %.cpp %.moc
	g++ $(CXXFLAGS) $(INCPATH) -c -o $@ $<

%.moc: %.h %.cpp
	moc $(DEFINES) $(INCPATH) -i $< -o $@

clean:
	-rm $(TARGET) $(OBJECTS) $(MOCFILES)
