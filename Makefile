CC = g++
CFLAGS = -g -Wall -fopenmp -o3
SRCS = SaliencyMap/main.cpp Filter/Filter.cpp SaliencyMap/SaliencyMap.cpp
PROG = out

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

build: SaliencyMap/main.cpp
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

clean:
	rm -rf out