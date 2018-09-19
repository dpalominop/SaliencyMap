//============================================================================
// Name        : Filter.cpp
// Author      : Daniel Palomino
// Version     : 1.0
// Copyright   : GNU General Public License v3.0
// Description : Parallel Matrix Convolution Class
// Created on  : 07 set. 2018
//============================================================================

#include "Filter.h"


Filter::Filter() {
	/*cudaMalloc((void**)&dev_kernel, 5*sizeof(double));
	for(int i=0; i<5; i++){
		cudaMalloc((void**)&dev_kernel[i], 5*sizeof(double));
	}*/
}

Filter::~Filter() {
	/*for(int i=0; i<5; i++){
		cudaFree(dev_kernel[i]);
	}
	cudaFree(dev_kernel);*/
}

Filter::Filter(double kernel[5*5]) {
	/*cudaMalloc((void**)&dev_kernel, 5*sizeof(double));
	for(int i=0; i<5; i++){
		cudaMalloc((void**)&dev_kernel[i], 5*sizeof(double));
	}*/
	setKernel(kernel);
}

Filter::Filter(double* kernel, int n) {
	/*cudaMalloc((void**)&dev_kernel, 5*sizeof(double));
	for(int i=0; i<5; i++){
		cudaMalloc((void**)&dev_kernel[i], 5*sizeof(double));
	}*/
	setKernel(kernel, 5);
}

bool Filter::setKernel(double kernel[5*5]) {
	setConvolutionKernel2(kernel);
	return true;
}

bool Filter::setKernel(double* kernel, int n) {
	setConvolutionKernel(kernel);
	return true;
}

bool Filter::convolution(double* &image, double* &result, int x_length, int y_length, int step)
{
	convolutionGPU(image, result, x_length, y_length, step);
	return true;
}

bool Filter::reserveMemory(double** &matrix, int x, int y) {

	matrix = new double*[x];
	for (int i = 0; i < x; i++) {
		matrix[i] = new double[y];
	}

	ASSERT(matrix != NULL);

	return true;
}

bool Filter::reserveMemory(double* &matrix, int x, int y) {

	matrix = new double[x*y];

	ASSERT(matrix != NULL);

	return true;
}

bool Filter::deleteMemory(double** &matrix, int x, int y) {
	for (int i = 0; i < x; i++) {
		delete[] matrix[i];
	}

	delete[] matrix;

	return true;
}

bool Filter::deleteMemory(double* &matrix, int x, int y) {

	delete[] matrix;

	return true;
}

bool Filter::generateData(double** &matrix, int x, int y) {
	srand(time(NULL));

	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			matrix[i][j] = rand() % 3;
		}
	}

	return false;
}

bool Filter::generateData(double* &matrix, int x, int y) {
	srand(time(NULL));

	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			matrix[i*y+j] = rand() % 3;
		}
	}

	return false;
}

bool Filter::showData(double** result, int x, int y) {
	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			std::cout << " " << result[i][j];
		}
		std::cout << std::endl;
	}

	return true;
}

bool Filter::showData(double* result, int x, int y) {
	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			std::cout << " " << result[i*y+j];
		}
		std::cout << std::endl;
	}

	return true;
}

bool Filter::showData(double result[5][5], int x, int y) {
	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			std::cout << " " << result[i][j];
		}
		std::cout << std::endl;
	}

	return true;
}

/*bool Filter::showKernel() {
	std::cout << "------------------" << std::endl;
	std::cout << "Kernel used: " << std::endl;

	for (int i = 0; i < dim_kernel; i++)
	{
		for (int j = 0; j < dim_kernel; j++)
		{
			std::cout << " " << h_kernel[j+i*dim_kernel];
		}
		std::cout << std::endl;
	}
	std::cout << "------------------" << std::endl;

	return true;
}*/

bool Filter::growthMatrix(double* matrix, double* result, int height, int width, int k) {

	growthMatrixGPU(matrix, result, height, width, k);

	return true;
}
