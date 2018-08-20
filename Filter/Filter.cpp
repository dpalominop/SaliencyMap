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

bool Filter::convolution(double** image, double** result, int i_length, int thread_count)
{
	int li_image = dim_kernel / 2;
	int ls_image = i_length - li_image;

	//Cuadrado central
#pragma omp parallel for collapse(2) num_threads(thread_count) shared(mkernel, image, result, thread_count)
	for (int i = 2; i < ls_image; i++) {
		for (int j = 2; j < ls_image; j++) {
			double acumulador = 0;
			double* krow;
			double* irow;

			krow = mkernel[0];
			irow = image[i - 2];
			acumulador += krow[0] * irow[j + -2];
			acumulador += krow[1] * irow[j + -1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + 1];
			acumulador += krow[4] * irow[j + 2];

			krow = mkernel[1];
			irow = image[i - 1];
			acumulador += krow[0] * irow[j + -2];
			acumulador += krow[1] * irow[j + -1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + 1];
			acumulador += krow[4] * irow[j + 2];

			krow = mkernel[2];
			irow = image[i];
			acumulador += krow[0] * irow[j + -2];
			acumulador += krow[1] * irow[j + -1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + 1];
			acumulador += krow[4] * irow[j + 2];

			krow = mkernel[3];
			irow = image[i + 1];
			acumulador += krow[0] * irow[j + -2];
			acumulador += krow[1] * irow[j + -1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + 1];
			acumulador += krow[4] * irow[j + 2];

			krow = mkernel[4];
			irow = image[i + 2];
			acumulador += krow[0] * irow[j + -2];
			acumulador += krow[1] * irow[j + -1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + 1];
			acumulador += krow[4] * irow[j + 2];

			result[i][j] = acumulador / 25;
		}
	}


	//Clculo de los bordes sin ezquinas
#pragma omp parallel sections
	{
#pragma omp section
		{
			double* krow;
			double* irow;
			for (int i = 2; i < ls_image; i++) {
				for (int j = 0; j < 2; j++) {
					double acumulador = 0;

					for (int m = 0; m < 5; m++) {
						krow = mkernel[m];
						irow = image[i + (m - 2)];

						if ((j - 2) >= 0) {
							acumulador += krow[0] * irow[j - 2];
						}
						if ((j - 1) >= 0) {
							acumulador += krow[1] * irow[j - 1];
						}
						acumulador += krow[2] * irow[j];
						acumulador += krow[3] * irow[j + 1];
						acumulador += krow[4] * irow[j + 2];
					}
					result[i][j] = acumulador / (15 + 5 * j);
				}
			}
		}

#pragma omp section
		{
			for (int i = 2; i < ls_image; i++) {
				for (int j = ls_image; j < i_length; j++) {
					double acumulador = 0;

					for (int m = 0; m < 5; m++) {
						double* krow = mkernel[m];
						double* irow = image[i + (m - 2)];

						acumulador += krow[0] * irow[j - 2];
						acumulador += krow[1] * irow[j - 1];
						acumulador += krow[2] * irow[j];
						if ((j + 1) < i_length) {
							acumulador += krow[3] * irow[j + 1];
						}
						if ((j + 2) < i_length) {
							acumulador += krow[4] * irow[j + 2];
						}
					}
					result[i][j] = acumulador / (15 + 5 * (i_length - 1 - j));
				}
			}
		}

#pragma omp section
		{
			for (int i = 0; i < 2; i++) {
				for (int j = 2; j < ls_image; j++) {
					int ii;
					double acumulador = 0;

					for (int m = 0; m < 5; m++) {
						double* krow = mkernel[m];
						ii = i + (m - 2);
						double* irow = image[ii];
						if (ii >= 0) {
							for (int n = 0; n < 5; n++) {
								acumulador += krow[n] * irow[j + (n - 2)];
							}
						}
					}
					result[i][j] = acumulador / (15 + 5 * i);
				}
			}
		}

#pragma omp section
		{
			for (int i = ls_image; i < i_length; i++) {
				for (int j = 2; j < ls_image; j++) {
					int ii;
					double acumulador = 0;

					for (int m = 0; m < 5; m++) {
						double* krow = mkernel[m];
						ii = i + (m - 2);
						double* irow = image[ii];
						if (ii < i_length) {
							for (int n = 0; n < 5; n++) {
								acumulador += krow[n] * irow[j + (n - 2)];
							}
						}
					}
					result[i][j] = acumulador / (15 + 5 * (i_length - 1 - i));
				}
			}
		}
	}

	//Clculo de ezquinas
#pragma omp parallel sections
	{
#pragma omp section
		{
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					int ii, jj;
					double acumulador = 0;
					int num = 0;

					for (int m = 0; m < 5; m++) {
						double* krow = mkernel[m];
						ii = i + (m - 2);
						double* irow = image[ii];
						for (int n = 0; n < 5; n++) {
							jj = j + (n - 2);
							if (ii >= 0 && jj >= 0) {
								acumulador += krow[n] * irow[jj];
								num++;
							}
						}
					}
					result[i][j] = acumulador / num;
				}
			}
		}
#pragma omp section
		{
			for (int i = ls_image; i < i_length; i++) {
				for (int j = 0; j < 2; j++) {
					int ii, jj;
					double acumulador = 0;
					int num = 0;

					for (int m = 0; m < 5; m++) {
						double* krow = mkernel[m];
						ii = i + (m - 2);
						double* irow = image[ii];
						for (int n = 0; n < 5; n++) {
							jj = j + (n - 2);
							if (ii < i_length && jj >= 0) {
								acumulador += krow[n] * irow[jj];
								num++;
							}
						}
					}
					result[i][j] = acumulador / num;
				}
			}
		}
#pragma omp section
		{
			for (int i = 0; i < 2; i++) {
				for (int j = ls_image; j < i_length; j++) {
					int ii, jj;
					double acumulador = 0;
					int num = 0;

					for (int m = 0; m < 5; m++) {
						double* krow = mkernel[m];
						ii = i + (m - 2);
						double* irow = image[ii];
						for (int n = 0; n < 5; n++) {
							jj = j + (n - 2);
							if (ii >= 0 && jj < i_length) {
								acumulador += krow[n] * irow[jj];
								num++;
							}
						}
					}
					result[i][j] = acumulador / num;
				}
			}
		}
#pragma omp section
		{
			for (int i = ls_image; i < i_length; i++) {
				for (int j = ls_image; j < i_length; j++) {
					int ii, jj;
					double acumulador = 0;
					int num = 0;

					for (int m = 0; m < 5; m++) {
						double* krow = mkernel[m];
						ii = i + (m - 2);
						double* irow = image[ii];
						for (int n = 0; n < 5; n++) {
							jj = j + (n - 2);
							if (ii < i_length && jj < i_length) {
								acumulador += krow[n] * irow[jj];
								num++;
							}
						}
					}
					result[i][j] = acumulador / num;
				}
			}
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