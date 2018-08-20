#pragma once

#include <math.h>

void minArray(double **arr, double &minimum, int rows, int cols) {
	int i, j;
	minimum = 99999999.999;

#   pragma omp for collapse(2) reduction(min:minimum)
	for (i = 0;i < rows;i++)
		for (j = 0;j < cols;j++)
			if (arr[i][j] < minimum)
				minimum = arr[i][j];
}


void maxArray(double **arr, double &maximum, int rows, int cols) {
	int i, j;
	maximum = -99999999.999;

#   pragma omp for collapse(2) reduction(min:minimum)
	for (i = 0;i < rows;i++)
		for (j = 0;j < cols;j++)
			if (arr[i][j] > maximum)
				maximum = arr[i][j];
}



void meanArray(double **arr, double &mean, int rows, int cols) {
	int i, j;
	mean = 0.0;

#   pragma omp for collapse(2) reduction(+:mean)
	for (i = 0;i < rows;i++)
		for (j = 0;j < cols;j++)
			mean += arr[i][j];

	mean = mean / (rows*cols);
}

void norm1Array(double **arr, double &norm, int rows, int cols) {
	int i, j;
	double sum;
	norm = -99999999.999;

#   pragma omp for reduction(max:norm)
	for (j = 0;j < cols;j++) {
		sum = 0.0;
		for (i = 0; i < rows;++i) {
			sum += arr[i][j];
		}
		if (sum > norm)
			norm = sum;
	}
}


void normInfArray(double **arr, double &norm, int rows, int cols) {
	int i, j;
	double sum;
	norm = -99999999.999;

#   pragma omp for reduction(max:norm)
	for (i = 0;i < rows; i++) {
		sum = 0.0;
		for (j = 0; j < cols; ++j) {
			sum += arr[i][j];
		}
		if (sum > norm)
			norm = sum;
	}
}


void norm2Array(double **arr, double &norm, int rows, int cols) {
	double norma1, normaInf;

	norm1Array(arr, norma1  , rows, cols);
	normInfArray(arr, normaInf, rows, cols);

	norm = sqrt(norma1*normaInf);
}

