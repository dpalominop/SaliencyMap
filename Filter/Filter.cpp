//============================================================================
// Name        : GaussianBlur.cpp
// Author      : Daniel Palomino
// Version     : 1.0
// Copyright   : GNU General Public License v3.0
// Description : Parallel Matrix Convolution Class
// Created on  : 19 ago. 2018
//============================================================================

#include "Filter.h"

Filter::Filter() {
	klength = dim_kernel;
	reserveMemory(mkernel, klength);
}

Filter::~Filter() {
	deleteMemory(mkernel, klength);
}

Filter::Filter(double kernel[][dim_kernel]) {
	ASSERT(length(kernel) == dim_kernel);

	klength = length(kernel);
	if (reserveMemory(mkernel, klength)) {
		setKernel(kernel);
		std::cout << "Kernel created" << std::endl;
	}
	else {
		std::cout << "Error in Kernel creation!" << std::endl;
	}
}

Filter::Filter(double** kernel, int n) {
	ASSERT(n == dim_kernel);

	klength = n;
	if (reserveMemory(mkernel, klength)) {
		setKernel(kernel, n);
		std::cout << "Kernel created" << std::endl;
	}
	else {
		std::cout << "Error in Kernel creation!" << std::endl;
	}
}

bool Filter::setKernel(double kernel[][dim_kernel]) {
	ASSERT(length(kernel) == dim_kernel);

	klength = length(kernel);
	for (int i = 0; i < klength; i++) {
		for (int j = 0; j < klength; j++) {
			mkernel[i][j] = kernel[i][j];
		}
	}
	return true;
}

bool Filter::setKernel(double** kernel, int n) {
	ASSERT(n == dim_kernel);

	klength = n;
	for (int i = 0; i < klength; i++) {
		for (int j = 0; j < klength; j++) {
			mkernel[i][j] = kernel[i][j];
		}
	}
	return true;
}

bool Filter::convolution(double** image, int i_length, double** result, int thread_count, int step)
{
	double** mImage;
	int mi_length = i_length + 2*(klength/2);
	reserveMemory(mImage, mi_length);

#pragma omp parallel for num_threads(thread_count) shared(mImage)
	for (int i = 0; i < mi_length; i++) {
		std::fill_n(mImage[i], mi_length, 0);
	}

	int li_mImage = klength / 2;
	int ls_mImage = mi_length - li_mImage;

#pragma omp parallel for collapse(2) num_threads(thread_count) shared(mImage, image)
	for (int i = li_mImage; i < ls_mImage; i++) {
		for (int j = li_mImage; j < ls_mImage; j++) {
			mImage[i][j] = image[i - 2][j - 2];
		}
	}

	//Cuadrado central
#pragma omp parallel for collapse(2) num_threads(thread_count) shared(mkernel, mImage, result, thread_count, step)
	for (int i = li_mImage; i < ls_mImage; i+=step) {
		for (int j = li_mImage; j < ls_mImage; j+= step) {
			double acumulador = 0;
			double* krow;
			double* irow;

			krow = mkernel[0];
			irow = mImage[i - 2];
			acumulador += krow[0] * irow[j + -2];
			acumulador += krow[1] * irow[j + -1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + 1];
			acumulador += krow[4] * irow[j + 2];

			krow = mkernel[1];
			irow = mImage[i - 1];
			acumulador += krow[0] * irow[j + -2];
			acumulador += krow[1] * irow[j + -1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + 1];
			acumulador += krow[4] * irow[j + 2];

			krow = mkernel[2];
			irow = mImage[i];
			acumulador += krow[0] * irow[j + -2];
			acumulador += krow[1] * irow[j + -1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + 1];
			acumulador += krow[4] * irow[j + 2];

			krow = mkernel[3];
			irow = mImage[i + 1];
			acumulador += krow[0] * irow[j + -2];
			acumulador += krow[1] * irow[j + -1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + 1];
			acumulador += krow[4] * irow[j + 2];

			krow = mkernel[4];
			irow = mImage[i + 2];
			acumulador += krow[0] * irow[j + -2];
			acumulador += krow[1] * irow[j + -1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + 1];
			acumulador += krow[4] * irow[j + 2];

			result[(i-2)/step][(j-2)/step] = acumulador / 25;
		}
	}

	return true;
}

bool Filter::reserveMemory(double** &matrix, int n) {

	matrix = new double*[n];
	for (int i = 0; i < n; i++) {
		matrix[i] = new double[n];
	}

	ASSERT(matrix != NULL);

	return true;
}

bool Filter::deleteMemory(double** &matrix, int n) {
	for (int i = 0; i < n; i++) {
		delete[] matrix[i];
	}

	delete[] matrix;

	return true;
}

bool Filter::generateData(double** &matrix, int n) {
	srand(time(NULL));

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			matrix[i][j] = rand() % 3;
		}
	}

	return false;
}

bool Filter::showData(double** result, int n) {
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			std::cout << " " << result[i][j];
		}
		std::cout << std::endl;
	}

	return true;
}

bool Filter::showKernel() {
	std::cout << "------------------" << std::endl;
	std::cout << "Kernel used: " << std::endl;

	for (int i = 0; i < klength; i++)
	{
		for (int j = 0; j < klength; j++)
		{
			std::cout << " " << mkernel[i][j];
		}
		std::cout << std::endl;
	}
	std::cout << "------------------" << std::endl;

	return true;
}