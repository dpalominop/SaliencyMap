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
	int klength = 5;

public:
	GaussianBlur(double[][] kernel);
	GaussianBlur(double** kernel);

	bool setKernel(double[][] kernel);
	bool setKernel(double** kernel);
	bool convolucion(double** image, double** result, int thread_count);
	bool showData(double** result, int n);
	bool generateData(double** matrix, int n);

	bool deleteMemory(double** &matrix, int n);
	bool reserveMemory(double** &matrix, int n);

};

#endif GAUSSIANBLUR_H