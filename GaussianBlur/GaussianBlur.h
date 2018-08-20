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
#define dim_image 2000
#define size_kernel dim_kernel*dim_kernel
#define li_image dim_kernel/2
#define ls_image dim_image-li_image


#define length(x) (sizeof(x)/sizeof(x[0]))

class GaussianBlur {

private:
	double** mkernel;
	int klength = dim_kernel;

public:
	GaussianBlur();
	GaussianBlur(double[][dim_kernel] kernel);
	GaussianBlur(double** kernel, int n);
	~GaussianBlur();

	bool setKernel(double[][dim_kernel] kernel);
	bool setKernel(double** kernel, int n);
	bool convolution(double** image, double** result, int thread_count);

//Static methods
public:

	static bool showData(double** result, int n);
	static bool generateData(double** &matrix, int n);

	static bool deleteMemory(double** &matrix, int n);
	static bool reserveMemory(double** &matrix, int n);

};

#endif