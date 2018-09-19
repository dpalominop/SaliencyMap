#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <omp.h>

static void minArray(double **arr, double &minimum, int rows, int cols, int thread_count) {
	int i, j;
	minimum = 99999999.999;

#   pragma omp parallel for collapse(2) num_threads(thread_count) reduction(min:minimum)
	for (i = 0;i < rows;i++)
		for (j = 0;j < cols;j++)
			if (arr[i][j] < minimum)
				minimum = arr[i][j];
}

static void maxArray(double **arr, double &maximum, int rows, int cols, int thread_count) {
	int i, j;
	maximum = -99999999.999;

#   pragma omp parallel for collapse(2) num_threads(thread_count) reduction(max:maximum)
	for (i = 0;i < rows;i++)
		for (j = 0;j < cols;j++)
			if (arr[i][j] > maximum)
				maximum = arr[i][j];
}


static void minArray(double *arr, double &minimum, int size, int thread_count) {
	int i;
	minimum = 99999999.999;

#   pragma omp parallel for num_threads(thread_count) reduction(min:minimum)
	for (i = 0;i < size; i++)
		if (arr[i] < minimum)
			minimum = arr[i];
}

static void maxArray(double *arr, double &maximum, int size, int thread_count) {
	int i;
	maximum = -99999999.999;

#   pragma omp parallel for num_threads(thread_count) reduction(max:maximum)
	for (i = 0;i < size;i++)
		if (arr[i] > maximum)
			maximum = arr[i];
}


static void meanArray(double **arr, double &mean, int rows, int cols, int thread_count) {
	int i, j;
	mean = 0.0;

#   pragma omp parallel for collapse(2) num_threads(thread_count) reduction(+:mean)
	for (i = 0;i < rows;i++)
		for (j = 0;j < cols;j++)
			mean += arr[i][j];

	mean = mean / (rows*cols);
}

static void norm1Array(double **arr, double &norm, int rows, int cols, int thread_count) {
	int i, j;
	double sum;
	norm = -99999999.999;

#   pragma omp parallel for num_threads(thread_count) reduction(max:norm)
	for (j = 0;j < cols;j++) {
		sum = 0.0;
		for (i = 0; i < rows;++i) {
			sum += abs(arr[i][j]);
		}
		if (sum > norm)
			norm = sum;
	}
}


static void normInfArray(double **arr, double &norm, int rows, int cols, int thread_count) {
	int i, j;
	double sum;
	norm = -99999999.999;

#   pragma omp parallel for num_threads(thread_count) reduction(max:norm)
	for (i = 0;i < rows; i++) {
		sum = 0.0;
		for (j = 0; j < cols; ++j) {
			sum += abs(arr[i][j]);
		}
		if (sum > norm)
			norm = sum;
	}
}


static void norm2Array(double **arr, double &norm, int rows, int cols, int thread_count) {
	double norma1, normaInf;

	norm1Array(arr, norma1  , rows, cols, thread_count);
	normInfArray(arr, normaInf, rows, cols, thread_count);

	norm = sqrt(norma1*normaInf);
}


static void nrm(double** &arr, int rows, int cols, int thread_count){
	double _norm, _max, _mean;
	double coeff;

	normInfArray(arr, _norm, rows, cols, thread_count);

	coeff = 1/_norm;

#pragma omp parallel for collapse(2) num_threads(thread_count) shared(coeff)
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			arr[i][j] = coeff*arr[i][j];
		}
	}

	 maxArray(arr, _max , rows, cols, thread_count);
	meanArray(arr, _mean, rows, cols, thread_count);

	coeff = (_max - _mean)*(_max - _mean);

#pragma omp parallel for collapse(2) num_threads(thread_count) shared(coeff)
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			arr[i][j] = coeff*arr[i][j];
		}
	}

	
}


static int pow2(int pot) {
	int out = 1;
	for (int i = 0; i < pot; ++i) out *= 2;
	return out;
}


struct Pyramid {
	double **_Base;
	double **_Level1;
	double **_Level2;
	double **_Level3;
	double **_Level4;
	double **_Level5;
	double **_Level6;
	double **_Level7;
	double **_Level8;
	int rows;
	int cols;

	Pyramid(int _rows, int _cols) : rows(_rows), cols(_cols) {
		int c;

		_Base = new double*[rows];
		for (int i = 0; i < rows; ++i) _Base[i] = new double[rows];

		c = 2; _Level1 = new double*[rows / c];
		for (int i = 0; i < rows / c; ++i) _Level1[i] = new double[rows / c];

		c = 4; _Level2 = new double*[rows / c];
		for (int i = 0; i < rows / c; ++i) _Level2[i] = new double[rows / c];

		c = 8; _Level3 = new double*[rows / c];
		for (int i = 0; i < rows / c; ++i) _Level3[i] = new double[rows / c];

		c = 16; _Level4 = new double*[rows / c];
		for (int i = 0; i < rows / c; ++i) _Level4[i] = new double[rows / c];

		c = 32; _Level5 = new double*[rows / c];
		for (int i = 0; i < rows / c; ++i) _Level5[i] = new double[rows / c];

		c = 64; _Level6 = new double*[rows / c];
		for (int i = 0; i < rows / c; ++i) _Level6[i] = new double[rows / c];

		c = 128; _Level7 = new double*[rows / c];
		for (int i = 0; i < rows / c; ++i) _Level7[i] = new double[rows / c];

		c = 256; _Level8 = new double*[rows / c];
		for (int i = 0; i < rows / c; ++i) _Level8[i] = new double[rows / c];
	}

	void clean() {
		int c;

		for (int i = 0; i < rows; ++i) delete[] _Base[i];
		delete[] _Base;

		c = 2; for (int i = 0; i < rows / c; ++i) delete[] _Level1[i];
		delete[] _Level1;

		c = 4; for (int i = 0; i < rows / c; ++i) delete[] _Level2[i];
		delete[] _Level2;

		c = 8; for (int i = 0; i < rows / c; ++i) delete[] _Level3[i];
		delete[] _Level3;

		c = 16; for (int i = 0; i < rows / c; ++i) delete[] _Level4[i];
		delete[] _Level4;

		c = 32; for (int i = 0; i < rows / c; ++i) delete[] _Level5[i];
		delete[] _Level5;

		c = 64; for (int i = 0; i < rows / c; ++i) delete[] _Level6[i];
		delete[] _Level6;

		c = 128; for (int i = 0; i < rows / c; ++i) delete[] _Level7[i];
		delete[] _Level7;

		c = 256; for (int i = 0; i < rows / c; ++i) delete[] _Level8[i];
		delete[] _Level8;

	}
};

#endif