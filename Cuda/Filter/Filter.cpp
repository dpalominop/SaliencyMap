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
	klength = dim_kernel;
	h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
}

Filter::~Filter() {
	deleteMemory(h_kernel, klength, klength);
}

Filter::Filter(const double kernel[][dim_kernel]) {
	h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
	setKernel(kernel, n);
}

Filter::Filter(double** kernel, int n) {
	h_Kernel = (float *)malloc(kernelH * kernelW * sizeof(float));
	setKernel(kernel, n);
}

bool Filter::setKernel(const double kernel[][dim_kernel]) {
	for (int i = 0; i < dim_kernel; i++) {
		for (int j = 0; j < dim_kernel; j++) {
			h_kernel[j+i*dim_kernel] = kernel[i][j];
		}
	}
	return true;
}

bool Filter::setKernel(double** kernel, int n) {
	for (int i = 0; i < dim_kernel; i++) {
		for (int j = 0; j < dim_kernel; j++) {
			h_kernel[j+i*dim_kernel] = kernel[i][j];
		}
	}
	return true;
}

bool Filter::convolution(double** image, int x_length, int y_length, double** result, int step)
{
	float
	*h_Data,
	*h_ResultCPU,
	*h_ResultGPU;

	float
	*d_Data,
	*d_PaddedData,
	*d_Kernel,
	*d_PaddedKernel;

	fComplex
	*d_DataSpectrum,
	*d_KernelSpectrum;

	cufftHandle
	fftPlanFwd,
	fftPlanInv;

	bool bRetVal;
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	printf("Testing built-in R2C / C2R FFT-based convolution\n");

	const int   dataH = y_length;
	const int   dataW = x_length;
	const int    fftH = snapTransformSize(dataH + kernelH - 1);
	const int    fftW = snapTransformSize(dataW + kernelW - 1);

	printf("...allocating memory\n");
	h_Data      = (float *)malloc(dataH   * dataW * sizeof(float));

	h_ResultCPU = (float *)malloc(dataH   * dataW * sizeof(float));
	h_ResultGPU = (float *)malloc(fftH    * fftW * sizeof(float));

	checkCudaErrors(cudaMalloc((void **)&d_Data,   dataH   * dataW   * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum,   fftH * (fftW / 2 + 1) * sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
	checkCudaErrors(cudaMemset(d_KernelSpectrum, 0, fftH * (fftW / 2 + 1) * sizeof(fComplex)));

	printf("...generating random input data\n");
	srand(2010);

	for (int i = 0; i < dataH; i++)
	{
		for (int j = 0; j < dataW; j++){
			h_Data[j+i*dataW] = image[i][j];
		}
	}

	printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
	checkCudaErrors(cufftPlan2d(&fftPlanFwd, fftH, fftW, CUFFT_R2C));
	checkCudaErrors(cufftPlan2d(&fftPlanInv, fftH, fftW, CUFFT_C2R));

	printf("...uploading to GPU and padding convolution kernel and input data\n");
	checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Data,   h_Data,   dataH   * dataW *   sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
	checkCudaErrors(cudaMemset(d_PaddedData,   0, fftH * fftW * sizeof(float)));

	padKernel(
		d_PaddedKernel,
		d_Kernel,
		fftH,
		fftW,
		kernelH,
		kernelW,
		kernelY,
		kernelX
	);

	padDataClampToBorder(
		d_PaddedData,
		d_Data,
		fftH,
		fftW,
		dataH,
		dataW,
		kernelH,
		kernelW,
		kernelY,
		kernelX
	);

	//Not including kernel transformation into time measurement,
	//since convolution kernel is not changed very frequently
	printf("...transforming convolution kernel\n");
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum));

	printf("...running GPU FFT convolution: ");
	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedData, (cufftComplex *)d_DataSpectrum));
	modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
	checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_DataSpectrum, (cufftReal *)d_PaddedData));

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	double gpuTime = sdkGetTimerValue(&hTimer);
	printf("%f MPix/s (%f ms)\n", (double)dataH * (double)dataW * 1e-6 / (gpuTime * 0.001), gpuTime);

	printf("...reading back GPU convolution results\n");
	checkCudaErrors(cudaMemcpy(h_ResultGPU, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));

	printf("...running reference CPU convolution\n");
	convolutionClampToBorderCPU(
		h_ResultCPU,
		h_Data,
		h_Kernel,
		dataH,
		dataW,
		kernelH,
		kernelW,
		kernelY,
		kernelX
	);

	printf("...comparing the results: ");
	double sum_delta2 = 0;
	double sum_ref2   = 0;
	double max_delta_ref = 0;

	for (int y = 0; y < dataH; y++)
		for (int x = 0; x < dataW; x++)
		{
			double  rCPU = (double)h_ResultCPU[y * dataW + x];
			double  rGPU = (double)h_ResultGPU[y * fftW  + x];
			double delta = (rCPU - rGPU) * (rCPU - rGPU);
			double   ref = rCPU * rCPU + rCPU * rCPU;

			if ((delta / ref) > max_delta_ref)
			{
				max_delta_ref = delta / ref;
			}

			sum_delta2 += delta;
			sum_ref2   += ref;
		}

	double L2norm = sqrt(sum_delta2 / sum_ref2);
	printf("rel L2 = %E (max delta = %E)\n", L2norm, sqrt(max_delta_ref));
	bRetVal = (L2norm < 1e-6) ? true : false;
	printf(bRetVal ? "L2norm Error OK\n" : "L2norm Error too high!\n");

	printf("...shutting down\n");
	sdkDeleteTimer(&hTimer);

	checkCudaErrors(cufftDestroy(fftPlanInv));
	checkCudaErrors(cufftDestroy(fftPlanFwd));

	checkCudaErrors(cudaFree(d_DataSpectrum));
	checkCudaErrors(cudaFree(d_KernelSpectrum));
	checkCudaErrors(cudaFree(d_PaddedData));
	checkCudaErrors(cudaFree(d_PaddedKernel));
	checkCudaErrors(cudaFree(d_Data));
	checkCudaErrors(cudaFree(d_Kernel));

	free(h_ResultGPU);
	free(h_ResultCPU);
	free(h_Data);
	free(h_Kernel);

	return bRetVal;

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

bool Filter::showKernel() {
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
}

bool Filter::growthMatrix(double** matrix, int x, int y, double** result, int k, int thread_count) {
#pragma omp parallel for collapse(2) num_threads(thread_count) shared(matrix, result)
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y-1; j++) {
			for (int p = 0; p < k; p++) {
				result[i*k][j*k + p] = matrix[i][j] + p * ((matrix[i][j + 1] - matrix[i][j]) / k);
			}
		}
	}

#pragma omp parallel for num_threads(thread_count) shared(matrix, result)
	for (int i = 0; i < x; i++) {
		for (int p = 0; p < k; p++) {
			result[i*k][(y-1)*k + p] = matrix[i][y-1] + p * ((matrix[i][y - 1] - matrix[i][y - 2])/k);
		}
	}

#pragma omp parallel for collapse(2) num_threads(thread_count) shared(matrix, result)
	for (int i = 0; i < x-1; i++) {
		for (int p = 1; p < k; p++) {
			for (int j = 0; j < y*k; j ++) {
				result[i*k+p][j] = matrix[i][j/k] + p * ((matrix[i+1][j/k] - matrix[i][j/k])/k);
			}
		}
	}

#pragma omp parallel for num_threads(thread_count) shared(matrix, result)
	for (int p = 1; p < k; p++) {
		for (int j = 0; j < y*k; j++) {
			result[(x-1)*k + p][j] = matrix[x-1][j/k] + p * ((matrix[x-1][j/k] - matrix[x-2][j/k])/k);
		}
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
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
)
{
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int borderH = dataH + kernelY;
    const int borderW = dataW + kernelX;

    if (y < fftH && x < fftW)
    {
        int dy, dx;

        if (y < dataH)
        {
            dy = y;
        }

        if (x < dataW)
        {
            dx = x;
        }

        if (y >= dataH && y < borderH)
        {
            dy = dataH - 1;
        }

        if (x >= dataW && x < borderW)
        {
            dx = dataW - 1;
        }

        if (y >= borderH)
        {
            dy = 0;
        }

        if (x >= borderW)
        {
            dx = 0;
        }

        d_Dst[y * fftW + x] = LOAD_FLOAT(dy * dataW + dx);
    }
}
