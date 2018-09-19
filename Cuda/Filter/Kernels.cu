// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "Constants.h"

__constant__ float dev_kernel[KERNEL_LENGTH][KERNEL_LENGTH];

extern "C" void setConvolutionKernel(double** h_Kernel)
{
    for (int i = 0; i < 5; i++) {
    	cudaMemcpyToSymbol(dev_kernel[i], h_Kernel[i], KERNEL_LENGTH*sizeof(double));
	}
}

extern "C" void setConvolutionKernel2(double h_Kernel[KERNEL_LENGTH][KERNEL_LENGTH])
{
    for (int i = 0; i < 5; i++) {
    	cudaMemcpyToSymbol(dev_kernel[i], h_Kernel[i], KERNEL_LENGTH*sizeof(double));
	}
}

__global__ void runConvolutionGPU(double** image, double** result, int height, int width, int step)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	//int O_TILE_WIDTH = blockDim.x-(lkernel/2)*2;
	//int O_TILE_HEIGHT = blockDim.y-(lkernel/2)*2;
	int row_o = threadIdx.y + blockIdx.y*O_TILE_HEIGHT;
	int col_o = threadIdx.x + blockIdx.x*O_TILE_WIDTH;

	int row_i = row_o - KERNEL_LENGTH/2;
	int col_i = col_o - KERNEL_LENGTH/2;

	__shared__ float N_ds[O_TILE_HEIGHT+(KERNEL_LENGTH/2)*2][O_TILE_WIDTH+(KERNEL_LENGTH/2)*2];

	if((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < height)){
		N_ds[ty][tx] = image[row_i][col_i];
	}else{
		N_ds[ty][tx] = 0.0f;
	}

	__syncthreads();

	double output = 0.0f;

	if(ty < O_TILE_HEIGHT && tx < O_TILE_WIDTH){
		for(int i=0; i<KERNEL_LENGTH; i++){
			for(int j=0; j<KERNEL_LENGTH; j++){
				output += dev_kernel[i][j]*N_ds[i+ty][i+tx];
			}
		}

		if(row_o < height && col_o < width){
			result[row_o][col_o] = output;
		}
	}
}

extern "C" void convolutionGPU(double** image, double** result, int x_length, int y_length, int step)
{
	double** dev_image, **dev_result;
	dim3 blocks(y_length/16 + (((y_length%16)==0)?0:1), x_length/16 + (((y_length%16)==0)?0:1));
	dim3 threads(16,16);

	cudaMalloc((void**)&dev_image, y_length*sizeof(double));
	cudaMalloc((void**)&dev_result, y_length*sizeof(double));
	for(int i=0; i<y_length; i++){
		cudaMalloc((void**)&dev_image[i], x_length*sizeof(double));
		cudaMemcpy(dev_image[i], image[i], x_length*sizeof(double), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_result[i], x_length*sizeof(double));
	}

	runConvolutionGPU<<<blocks,threads>>>(dev_image, dev_result, y_length, x_length, step);

	for(int i=0; i<y_length; i++){
		cudaMemcpy(result[i], dev_result[i], x_length*sizeof(double), cudaMemcpyDeviceToHost);
	}

	for(int i=0; i<y_length; i++){
		cudaFree(dev_image[i]);
		cudaFree(dev_result[i]);
	}
	cudaFree(dev_image);
	cudaFree(dev_result);
}
