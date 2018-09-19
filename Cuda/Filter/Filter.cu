

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>
#include "Constants.h"
#include "Filter.h"

__constant__ double dev_kernel[KERNEL_LENGTH*KERNEL_LENGTH];

extern "C" void setConvolutionKernel(double* h_Kernel)
{
    //for (int i = 0; i < KERNEL_LENGTH; i++) {
	cudaMemcpyToSymbol(dev_kernel, h_Kernel, KERNEL_LENGTH*KERNEL_LENGTH*sizeof(double));
	//}
}

extern "C" void setConvolutionKernel2(double h_Kernel[KERNEL_LENGTH*KERNEL_LENGTH])
{
    //for (int i = 0; i < KERNEL_LENGTH; i++) {
	cudaMemcpyToSymbol(dev_kernel, h_Kernel, KERNEL_LENGTH*KERNEL_LENGTH*sizeof(double));
	//}
}

__global__ void runConvolutionGPU(double* image, double* result, int height, int width, int step)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	//int O_TILE_WIDTH = blockDim.x-(lkernel/2)*2;
	//int O_TILE_HEIGHT = blockDim.y-(lkernel/2)*2;
	int row_o = threadIdx.y + blockIdx.y*O_TILE_HEIGHT;
	int col_o = threadIdx.x + blockIdx.x*O_TILE_WIDTH;

	int row_i = row_o - KERNEL_LENGTH/2;
	int col_i = col_o - KERNEL_LENGTH/2;

	__shared__ double N_ds[BLOCK_DIM_Y][BLOCK_DIM_X];

	if((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < height)){
		N_ds[ty][tx] = image[row_i*width+col_i];
	}else{
		N_ds[ty][tx] = 0.0f;
	}

	__syncthreads();

	double output = 0.0f;
	if(tx%step ==0 && ty%step==0 && ty < O_TILE_HEIGHT && tx < O_TILE_WIDTH){
		for(int i=0; i<KERNEL_LENGTH; i++){
			for(int j=0; j<KERNEL_LENGTH; j++){
				output += dev_kernel[i*KERNEL_LENGTH+j]*N_ds[(i+ty)][(j+tx)];
			}
		}
		if(row_o < height && col_o < width){
			result[(row_o/step)*width/step+col_o/step] = output;
		}
	}
}

extern "C" void convolutionGPU(double* image, double* result, int x_length, int y_length, int step)
{
	/*double* dev_image, *dev_result;

	cudaMalloc((void**)&dev_image, x_length*y_length*sizeof(double));
	cudaMalloc((void**)&dev_result, x_length*y_length*sizeof(double));

	cudaMemcpy(dev_image, image, x_length*y_length*sizeof(double), cudaMemcpyHostToDevice);
*/
	dim3 blocks(y_length/O_TILE_HEIGHT + (((y_length%O_TILE_HEIGHT)==0)?0:1), x_length/O_TILE_WIDTH + (((y_length%O_TILE_WIDTH)==0)?0:1));
	dim3 threads(BLOCK_DIM_Y,BLOCK_DIM_X);
	runConvolutionGPU<<<blocks,threads>>>(image, result, y_length, x_length, step);
/*
	cudaMemcpy(result, dev_result, x_length*y_length*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_image);
	cudaFree(dev_result);*/
}

__global__ void growthMatrixCol(double* matrix, double* result, int height, int width, int k) {

	int ty = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x + blockIdx.x*blockDim.x;

	if(ty < height && tx < width){
		if(tx < (width-1)){
			for (int p = 0; p < k; p++) {
				result[ty*k*width*k + tx*k + p] = matrix[ty*width + tx] + p * ((matrix[ty*width + tx + 1] - matrix[ty*width + tx]) / k);
			}
		}else{
			for (int p = 0; p < k; p++) {
				result[ty*k*width*k + (width-1)*k + p] = matrix[ty*width + width-1] + p * ((matrix[ty*width + width - 1] - matrix[ty*width + width - 2])/k);
			}
		}
	}
}

__global__ void growthMatrixRow(double* matrix, double* result, int height, int width, int k) {

	int ty = threadIdx.y + blockIdx.y*blockDim.y;
	int tx = threadIdx.x + blockIdx.x*blockDim.x;

	if(ty < height && tx < width){
		if(ty < (height-1)){
			for (int p = 1; p < k; p++) {
				result[(ty*k+p)*width + tx] = matrix[ty*width + tx/k] + p * ((matrix[(ty+1)*width + tx/k] - matrix[ty*width + tx/k])/k);
			}

		}else{
			for (int p = 1; p < k; p++) {
				result[((height-1)*k + p)*width + tx] = matrix[(height-1)*width + tx/k] + p * ((matrix[(height-1)*width + tx/k] - matrix[(height-2)*width + tx/k])/k);
			}
		}
	}
}

extern "C" void growthMatrixGPU(double* matrix, double* result, int height, int width, int k){

	double *dev_matrix, *dev_result, *dev_result_t;

	cudaMalloc((void**)&dev_matrix, height*width*sizeof(double));
	cudaMalloc((void**)&dev_result, height*k*width*k*sizeof(double));
	cudaMalloc((void**)&dev_result_t, height*k*width*k*sizeof(double));

	cudaMemcpy(dev_matrix, matrix, height*width*sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimBlock(8, 8);
    dim3 dimGrid(height/8 + (((height%8)==0)?0:1),width/8 + (((width%8)==0)?0:1));

    growthMatrixCol<<<dimGrid,dimBlock>>>(dev_matrix, dev_result, height, width, k);

    //dim3 dimGrid2(height/16 + (((height%16)==0)?0:1),width*k/16 + ((((width*k)%16)==0)?0:1));
    //growthMatrixRow<<<dimGrid2,dimBlock>>>(dev_result, dev_result_t, height, width*k, k);

    cudaMemcpy(result, dev_result, height*k*width*k*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_matrix);
	cudaFree(dev_result);
	cudaFree(dev_result_t);
}

