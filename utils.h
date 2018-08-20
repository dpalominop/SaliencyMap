
#ifndef UTILS_H
#define UTILS_H

#include <math.h>

void minArray(double **arr, double &minimum, int rows, int cols, int thread_count) {
	int i, j;
	minimum = 99999999.999;

//#   pragma omp parallel for collapse(2) num_threads(thread_count) reduction(min:minimum)
	for (i = 0;i < rows;i++)
		for (j = 0;j < cols;j++)
			if (arr[i][j] < minimum)
				minimum = arr[i][j];
}

void maxArray(double **arr, double &maximum, int rows, int cols, int thread_count) {
	int i, j;
	maximum = -99999999.999;

//#   pragma omp parallel for collapse(2) num_threads(thread_count) reduction(max:maximum)
	for (i = 0;i < rows;i++)
		for (j = 0;j < cols;j++)
			if (arr[i][j] > maximum)
				maximum = arr[i][j];
}

void meanArray(double **arr, double &mean, int rows, int cols, int thread_count) {
	int i, j;
	mean = 0.0;

//#   pragma omp parallel for collapse(2) num_threads(thread_count) reduction(+:mean)
	for (i = 0;i < rows;i++)
		for (j = 0;j < cols;j++)
			mean += arr[i][j];

	mean = mean / (rows*cols);
}

void norm1Array(double **arr, double &norm, int rows, int cols, int thread_count) {
	int i, j;
	double sum;
	norm = -99999999.999;

//#   pragma omp parallel for num_threads(thread_count) reduction(max:norm)
	for (j = 0;j < cols;j++) {
		sum = 0.0;
		for (i = 0; i < rows;++i) {
			sum += abs(arr[i][j]);
		}
		if (sum > norm)
			norm = sum;
	}
}


void normInfArray(double **arr, double &norm, int rows, int cols, int thread_count) {
	int i, j;
	double sum;
	norm = -99999999.999;

//#   pragma omp parallel for num_threads(thread_count) reduction(max:norm)
	for (i = 0;i < rows; i++) {
		sum = 0.0;
		for (j = 0; j < cols; ++j) {
			sum += abs(arr[i][j]);
		}
		if (sum > norm)
			norm = sum;
	}
}


void norm2Array(double **arr, double &norm, int rows, int cols, int thread_count) {
	double norma1, normaInf;

	norm1Array(arr, norma1  , rows, cols, thread_count);
	normInfArray(arr, normaInf, rows, cols, thread_count);

	norm = sqrt(norma1*normaInf);
}


bool interpolation(double** matrix, int x, int y, double** &result, int k, int thread_count) {
#pragma omp parallel for collapse(2) num_threads(thread_count) shared(matrix, result)
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y-1; j++) {
			double factor = (matrix[i][j + 1] - matrix[i][j]) / k;
			for (int p = 0; p < k; p++) {
				result[i*k][j*k + p] = matrix[i][j] + p * factor;
			}
		}
	}

#pragma omp parallel for collapse(2) num_threads(thread_count) shared(matrix, result)
	for (int i = 0; i < x; i++) {
		for (int p = 1; p < k; p++) {
			result[i*k][(y-1)*k + p] = matrix[i][y-1] + p * (matrix[i][y - 1] - matrix[i][y - 2]) / k;
		}
	}

#pragma omp parallel for collapse(2) num_threads(thread_count) shared(matrix, result)
	for (int i = 0; i < x-1; i++) {
		for (int p = 0; p < k; p++) {
			for (int j = 0; j < y; j ++) {
				result[i*k+p][j*k] = matrix[i][j] + p * (matrix[i+1][j] - matrix[i][j])/k;
			}
		}
	}

#pragma omp parallel for collapse(2) num_threads(thread_count) shared(matrix, result)
	for (int p = 0; p < k; p++) {
		for (int j = 0; j < y; j++) {
			result[(x-1)*k + p][j*k] = matrix[x-1][j] + p * (matrix[x-1][j] - matrix[x-2][j]) / k;
		}
	}

	return true;
}

#endif