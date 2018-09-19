//============================================================================
// Name        : FIlter.h
// Author      : Daniel Palomino
// Version     : 1.0
// Copyright   : GNU General Public License v3.0
// Description : Parallel Matrix Convolution Class
// Created on  : 06 set. 2018
//============================================================================

#ifndef FILTER_H
#define FILTER_H

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionSeparable_common.h"

#define ASSERT assert
#define dim_kernel 5
#define size_kernel dim_kernel*dim_kernel
#define length(x) (sizeof(x)/sizeof(x[0]))

#define  USE_TEXTURE 1
#define POWER_OF_TWO 1


#if(USE_TEXTURE)
texture<float, 1, cudaReadModeElementType> texFloat;
#define   LOAD_FLOAT(i) tex1Dfetch(texFloat, i)
#define  SET_FLOAT_BASE checkCudaErrors( cudaBindTexture(0, texFloat, d_Src) )
#else
#define  LOAD_FLOAT(i) d_Src[i]
#define SET_FLOAT_BASE
#endif

class Filter {

private:
	double* h_kernel;

	const int kernelH = 5;
	const int kernelW = 5;
	const int kernelY = 3;
	const int kernelX = 3;

public:
	Filter();
	Filter(const double kernel[][dim_kernel]);
	Filter(double** kernel, int n);
	~Filter();

	bool setKernel(const double kernel[][dim_kernel]);
	bool setKernel(double** kernel, int n);
	bool showKernel();
	bool convolution(double** image, int x_length, int y_length, double** result, int step);

private:
	__global__ void padKernel_kernel(
	    float *d_Dst,
	    float *d_Src,
	    int fftH,
	    int fftW,
	    int kernelH,
	    int kernelW,
	    int kernelY,
	    int kernelX
	);
	__global__ void padDataClampToBorder_kernel(
	    float *d_Dst,
	    float *d_Src,
	    int fftH,
	    int fftW,
	    int dataH,
	    int dataW,
	    int kernelH,
	    int kernelW,
	    int kernelY,
	    int kernelX
	);

//Static methods
public:

	static bool showData(double** result, int x, int y);
	static bool generateData(double** &matrix, int x, int y);

	static bool deleteMemory(double** &matrix, int x, int y);
	static bool reserveMemory(double** &matrix, int x, int y);
	static bool deleteMemory(double* &matrix, int x, int y);
	static bool reserveMemory(double* &matrix, int x, int y);

	static bool growthMatrix(double** matrix, int x, int y, double** result, int k, int thread_count);

};

#endif
