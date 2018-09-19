all: build

build: test.cpp
	mkdir -p out
	g++ -fopenmp -o out/test -std=c++11 -I. test.cpp Filter.cpp

exec: build
	./out/test

clean:
	rm -rf out
