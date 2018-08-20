//============================================================================
// Name        : GaussianBlur.h
// Author      : Daniel Palomino
// Version     : 1.0
// Copyright   : GNU General Public License v3.0
// Description : Parallel Matrix Convolution Class
// Created on  : 19 ago. 2018
//============================================================================

#ifndef GAUSSIANBLUR_H
#define GAUSSIANBLUR_H

#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>

#define ASSERT assert
#define dim_kernel 5
#define size_kernel dim_kernel*dim_kernel
#define length(x) (sizeof(x)/sizeof(x[0]))

class Filter {

private:
	double** mkernel;
	int klength = dim_kernel;

public:
	Filter();
	Filter(double kernel[][dim_kernel]);
	Filter(double** kernel, int n);
	~Filter();

	bool setKernel(double kernel[][dim_kernel]);
	bool setKernel(double** kernel, int n);
	bool showKernel();
	bool convolution(double** image, int i_length, double** result, int thread_count, int step);

//Static methods
public:

	static bool showData(double** result, int n);
	static bool generateData(double** &matrix, int n);

	static bool deleteMemory(double** &matrix, int n);
	static bool reserveMemory(double** &matrix, int n);

};

#endif