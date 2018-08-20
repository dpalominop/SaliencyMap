#pragma once

void minArray(double **arr, double &minimum, int rows, int cols) {
	int i, j;
	minimum = 99999999.999;

#   pragma omp for collapse(2) reduction(minimum:minimum)
	for (i = 0;i < rows;i++)
		for (j = 0;j < cols;j++)
			if (arr[i][j] < minimum)
				minimum = arr[i][j];
}

void meanArray(double **arr, double &mean, int rows, int cols) {
	int i, j;
	mean = 0.0;

#   pragma omp for collapse(2) reduction(mean:mean)
	for (i = 0;i < rows;i++)
		for (j = 0;j < cols;j++)
			mean += arr[i][j];

	mean = mean / (rows*cols);
}
