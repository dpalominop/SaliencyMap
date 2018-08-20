#include "Filter.h"
#define dim_image 7

using namespace std;

int main(int argc, char** argv) {

	Filter gb;

	int thread_count;
	bool show = false;

	if (argc == 2) {
		thread_count = strtol(argv[1], NULL, 10);
	}
	else {
		if (argc == 3) {
			thread_count = strtol(argv[1], NULL, 10);
			show = true;
		}
		else {
			return 0;
		}
	}

	double** kernel;
	double** image;
	double** result;

	kernel = new double*[dim_kernel];
	for (int i = 0; i < dim_kernel; i++) {
		kernel[i] = new double[dim_kernel];
	}

	image = new double*[dim_image];
	for (int i = 0; i < dim_image; i++) {
		image[i] = new double[dim_image];
	}

	result = new double*[dim_image];
	for (int i = 0; i < dim_image; i++) {
		result[i] = new double[dim_image];
	}

	Filter::generateData(kernel, dim_kernel);
	Filter::generateData(image, dim_image);

	//omp_set_nested (1);
	gb.setKernel(kernel, dim_kernel);
	double time_init = omp_get_wtime();
	gb.convolution(image, result, dim_image, thread_count);
	double time_final = omp_get_wtime();

	double n_elapsed = time_final - time_init;
	cout << "Minimal Total Time: " << n_elapsed << endl;

	if (show) {
		Filter::showData(kernel, dim_kernel);
		Filter::showData(image, dim_image);
		Filter::showData(result, dim_image);
	}

	for (int i = 0; i < dim_kernel; i++) {
		delete[] kernel[i];
	}

	delete[] kernel;

	for (int i = 0; i < dim_image; i++) {
		delete[] image[i];
	}

	delete[] image;

	for (int i = 0; i < dim_image; i++) {
		delete[] result[i];
	}

	delete[] result;

	return 0;
}