all: build

build: main.cpp
	mkdir -p out
	g++ -fopenmp -o out/gb -std=c++11 -I. main.cpp Filter.cpp

exec: build
	./out/gb

clean:
	rm -rf out
