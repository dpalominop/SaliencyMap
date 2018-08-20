#include "Filter.h"
#define dim_image 7

using namespace std;

int main(int argc, char** argv) {

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

	Filter gb;
	double time_init = 0;
	double time_final = 0;
	double** kernel;
	double** image;
	double** result;
	
	Filter::reserveMemory(kernel, dim_kernel);
	Filter::reserveMemory(image, dim_image);
	Filter::reserveMemory(result, dim_image);

	Filter::generateData(kernel, dim_kernel);
	Filter::generateData(image, dim_image);

	//omp_set_nested (1);
	gb.setKernel(kernel, dim_kernel);
	gb.showKernel();

	time_init = omp_get_wtime();
	gb.convolution(image, result, dim_image, thread_count);
	time_final = omp_get_wtime();

	cout << "Minimal Total Time: " << time_final - time_init << endl;

	if (show) {
		Filter::showData(kernel, dim_kernel);
		Filter::showData(image, dim_image);
		Filter::showData(result, dim_image);
	}

	Filter::deleteMemory(kernel, dim_kernel);
	Filter::deleteMemory(image, dim_image);
	Filter::deleteMemory(result, dim_image);

	return 0;
}