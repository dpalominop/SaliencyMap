CC = g++
CFLAGS = -g -Wall -fopenmp -o3
SRCS = main.cpp Filter/Filter.cpp utils.h
PROG = SaliencyMap

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

clean:
	rm -rf SaliencyMap