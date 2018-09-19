#include "SaliencyMap.h"
#include <cuda_runtime.h>

void gpuHostAlloc(double*& d_p, int rows, int cols) {
	//double* dPointer;
	cudaHostAlloc( (void**)&d_p, rows*cols*sizeof(double), cudaHostAllocMapped );
	//return dPointer;
}

void gpuMalloc(double*& d_p, int rows, int cols){
    cudaMalloc((void**)&d_p, rows*cols*sizeof(double));
}

void gpuFreeHostAlloc(double*& d_p){
    cudaFreeHost(d_p);
}

void gpuFreeMalloc(double*& d_p){
    cudaFree(d_p);
}


/*
 *  CPU Extra-functions
 *  ===================
 */
__global__ void matInfinityNorm(double *device_InMat,double *device_InfinityNorm,
                                int matRowSize, int matColSize, int threadDim){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tindex = (threadDim * tidx) + tidy;
    int maxNumThread = threadDim * threadDim; 
    int pass = 0;  
    int colCount, tCount ;
    int curRowInd;
    double tempInfinityNorm = 0.0;
    double rowMaxValue = 0.0;
      
    for( tCount = 1; tCount < maxNumThread; tCount++)
         device_InfinityNorm[tCount] = 0.0; 

    while( (curRowInd = (tindex + maxNumThread * pass))  < matRowSize ){
        rowMaxValue = 0.0;
        for( colCount = 0; colCount < matColSize; colCount++)
            rowMaxValue += abs(device_InMat[curRowInd* matRowSize + colCount]);
        tempInfinityNorm = ( tempInfinityNorm>rowMaxValue? tempInfinityNorm:rowMaxValue);
        pass++;
    }

    device_InfinityNorm[ tindex ] = tempInfinityNorm;
     __syncthreads();
   
    if(tindex == 0){
        for( tCount = 1; tCount < maxNumThread; tCount++)
            device_InfinityNorm[0] = device_InfinityNorm[0]> device_InfinityNorm[tCount]? device_InfinityNorm[0]: device_InfinityNorm[tCount]; 
    }
}


__global__ void meanMatrix(double *dMatrix, double *dMean, int dSize, int *d_mutex){
    __shared__ double cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    double temp = 0;
    while (tid < dSize) {
        temp += dMatrix[tid];
        tid  += blockDim.x * gridDim.x;
    }
    // set the cache values
    cache[cacheIndex] = temp;
    // synchronize threads in this block
    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if(cacheIndex == 0){
		while(atomicCAS(d_mutex,0,1) != 0);  //lock
		*dMean += cache[0];
        atomicExch(d_mutex, 0);  //unlock
        
        *dMean = dMean[0]/dSize;
	}
}


__global__ void find_maximum(double *array, double *max, int dSize, int *d_mutex){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = gridDim.x*blockDim.x;
	int offset = 0;

	__shared__ double cache[threadsPerBlock];

	double temp = -999999999.0;
	while(index + offset < dSize){
        temp = fmaxf(temp, array[index + offset]);
		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
            cache[threadIdx.x] = fmax(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

    if(threadIdx.x == 0){
		while(atomicCAS(d_mutex,0,1) != 0);  //lock
		*max = fmax(*max, cache[0]);
		atomicExch(d_mutex, 0);  //unlock
	}
}



__global__ void applyNormSum(double *dMap,double *dSupFeature, double *dMaxSupFeature, double *dMeanSupFeature,
                                          double *dInfFeature, double *dMaxInfFeature, double *dMeanInfFeature,
                                          int dSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    double SupCoeff = (dMaxSupFeature[0] - dMeanSupFeature[0])*(dMaxSupFeature[0] - dMeanSupFeature[0]);
    double InfCoeff = (dMaxInfFeature[0] - dMeanInfFeature[0])*(dMaxInfFeature[0] - dMeanInfFeature[0]);

    while (tid < dSize) {
        dMap[tid] += dSupFeature[tid]*SupCoeff + dInfFeature[tid]*InfCoeff;
        tid  += blockDim.x * gridDim.x;
    }
}

__global__ void absDifference(double *dDifference, double *dSup, double *dLow, int dSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < dSize) {
        double a = dSup[tid];
        double b = dLow[tid];
        dDifference[tid] = (a > b) ? (a - b) : (b - a);
        tid  += blockDim.x * gridDim.x;
    }
}

__global__ void sum3(double *d_result, 
                double *d_a, double *d_b, double *d_c, 
                int dSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < dSize) {
        d_result[tid] = d_a[tid] + d_b[tid] +d_c[tid];
        tid  += blockDim.x * gridDim.x;
    }
}

__global__ void divScalarMatrix(double *dMatrix, double *dScalar, int dSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < dSize) {
        dMatrix[tid] = dMatrix[tid]/dScalar[0];
        tid  += blockDim.x * gridDim.x;
    }
}

__global__ void kernelInterpolationRow(double *original, double *result, 
                                       int rows, int cols, int factor){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x +  y * blockDim.x * gridDim.x;

    int idOriginal,idResult;

    // Puntos de referencia para interpolacion
    double a,b;
    double   m; 

    //
    // Interpolacion de filas
    // ----------------------
    while (x < rows){
        idOriginal = y*rows               + x       ;
        idResult   = y*rows*factor*factor + x*factor;

        a = original[ idOriginal    ];
        b = original[ idOriginal + 1];

        m = (b - a)/((double)factor);

        // Antes de llegar al final
        if (x != rows-1){
            for(int p=0; p<=factor; ++p){
                result[idResult] = a;
                a += m;
                ++idResult;
            }
        }
        
        // Borde final
        else{
            for(int p=0; p<factor; ++p){
                result[idResult] = b;
                b -= m;
                ++idResult;
            }
        }

    }

}


__global__ void kernelInterpolationCol(double *result, 
                                       int rows, int cols, int factor){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x +  y * blockDim.x * gridDim.x;

    int idOriginal,idResult;

    // Puntos de referencia para interpolacion
    double a,b;
    double   m; 

    //
    // Interpolacion de columnas
    // -------------------------
    while (x < cols*factor && y<rows){
        int trueY = y*factor;
        int offset = x + trueY*cols*factor;

        a = result[ offset                     ];
        b = result[ offset + cols*factor*factor];

        m = (b - a)/((double)factor);

        // Antes de llegar al final
        if (y != rows-1){
            for(int p=0; p<=factor; ++p){
                result[offset] = a;
                a += m;
                offset += cols*factor*factor;
            }
        }
        
        // Borde final
        else{
            for(int p=0; p<factor; ++p){
                result[offset] = b;
                b -= m;
                offset += cols*factor*factor;
            }
        }
    }

}

void getMap(double* &feature, double* &map, 
                        const double kernel[][5],
                        int rows, int cols) {
    uint c;
    float dNorm25 = 0.0f, dNorm26 = 0.0f; 
    float dNorm36 = 0.0f, dNorm37 = 0.0f; 
    float dNorm47 = 0.0f, dNorm48 = 0.0f; 

    // Allocate Host-Device
    double *dFeature;
    double *dPyLevel1, *dPyLevel2;
    double *dPyLevel3, *dPyLevel4;
    double *dPyLevel5, *dPyLevel6;
    double *dPyLevel7, *dPyLevel8;

    double *feat25, *feat36, *feat47;

    c =   4*  4; cudaMalloc(&dPyLevel2, rows*cols/c*sizeof(double));
    c =   8*  8; cudaMalloc(&dPyLevel3, rows*cols/c*sizeof(double));
    c =  16* 16; cudaMalloc(&dPyLevel4, rows*cols/c*sizeof(double));
    c =  32* 32; cudaMalloc(&dPyLevel5, rows*cols/c*sizeof(double));
    c =  64* 64; cudaMalloc(&dPyLevel6, rows*cols/c*sizeof(double));
    c = 128*128; cudaMalloc(&dPyLevel7, rows*cols/c*sizeof(double));
    c = 256*256; cudaMalloc(&dPyLevel8, rows*cols/c*sizeof(double));

    c = 4*4; cudaMalloc(&feat25, rows*cols/c*sizeof(double));
    c = 4*4; cudaMalloc(&feat36, rows*cols/c*sizeof(double));
    c = 4*4; cudaMalloc(&feat47, rows*cols/c*sizeof(double));

    // Handles
    Filter blur(kernel);

    // Generate pyramid
    blur.convolution( feature , dPyLevel1, rows    , cols    , 2);
    blur.convolution(dPyLevel1, dPyLevel2, rows/2  , cols/2  , 2);
    blur.convolution(dPyLevel2, dPyLevel3, rows/4  , cols/4  , 2);
    blur.convolution(dPyLevel3, dPyLevel4, rows/8  , cols/8  , 2);
    blur.convolution(dPyLevel4, dPyLevel5, rows/16 , cols/16 , 2);
    blur.convolution(dPyLevel5, dPyLevel6, rows/32 , cols/32 , 2);
    blur.convolution(dPyLevel6, dPyLevel7, rows/64 , cols/64 , 2);
    blur.convolution(dPyLevel7, dPyLevel8, rows/128, cols/128, 2);

    // Center-surround difference
    centerSurroundDiff(dPyLevel2, dPyLevel5, feat25, 2, 5, 2);
    centerSurroundDiff(dPyLevel2, dPyLevel6, feat26, 2, 6, 2);

    centerSurroundDiff(dPyLevel3, dPyLevel6, feat36, 3, 6, 2);
    centerSurroundDiff(dPyLevel3, dPyLevel7, feat37, 3, 7, 2);

    centerSurroundDiff(dPyLevel4, dPyLevel7, feat47, 4, 7, 2);
    centerSurroundDiff(dPyLevel4, dPyLevel8, feat48, 4, 8, 2);

    // Free pyramid
    cudaFree(dPyLevel1); cudaFree(dPyLevel2);
    cudaFree(dPyLevel3); cudaFree(dPyLevel4);
    cudaFree(dPyLevel5); cudaFree(dPyLevel6);
    cudaFree(dPyLevel7); cudaFree(dPyLevel8);

    // Normalizarion
    c = 4*4;
    nrmSumGPU(feat25,feat26,map,rows*cols/c);
    nrmSumGPU(feat36,feat37,map,rows*cols/c);
    nrmSumGPU(feat47,feat48,map,rows*cols/c);

    // Free proto-feature
    cudaFree(feat25); cudaFree(feat26);
    cudaFree(feat36); cudaFree(feat37);
    cudaFree(feat47); cudaFree(feat48);
}

void centerSurroundDiffGPU(double* &dSupLevel, double* &dLowLevel,
                           double* &dDifference, 
                           int sup, int low, int endl,
                           int rows, int cols){
    int supRow = rows / pow2(sup);
    int supCol = cols / pow2(sup);

    int lowRow = rows / pow2(low);
    int lowCol = cols / pow2(low);

    // Interpolation
    double* dLowLevelGrownUp;
    cudaMalloc(&dLowLevelGrownUp, supRow*supCol*sizeof(double));
    Filter::growthMatrix(dLowLevel, lowRow, lowCol, 
        dLowLevelGrownUp, pow2(low - sup));

    if (sup != endl) {
        double* dRawDifference;
        cudaMalloc(&dRawDifference, supRow*supCol*sizeof(double));

        absDifference<<<BlocksInGrid,threadsPerBlock>>>(dRawDifference, dSupLevel, 
                                     dLowLevelGrownUp, supRow*supCol);
        Filter::growthMatrix(dRawDifference, supRow, supCol, 
                dDifference, pow2(sup - endl));

        cudaFree(dRawDifference);
    }
    else {
        absDifference<<<BlocksInGrid,threadsPerBlock>>>(dDifference, dSupLevel, 
                                     dLowLevelGrownUp, supRow*supCol);
    }

    // Liberar memoria
    cudaFree(dLowLevelGrownUp);
}


void nrmSumGPU(double* &dProSupFeature, double* &dProInfFeature, 
               double* &dMap,
               int rows, int cols){
    //
    // Calculo de norma infinito
    // -------------------------

    // Separar memoria
    double *dInfNormProSupFeature, *dInfNormProInfFeature;
    double *dMaxProSupFeature , *dMaxProInfFeature;
    double *dMeanProSupFeature, *dMeanProInfFeature;

    cudaMalloc(&dInfNormProSupFeature, sizeof(double)*BLOCKSIZE*BLOCKSIZE);
    cudaMalloc(&dInfNormProInfFeature, sizeof(double)*BLOCKSIZE*BLOCKSIZE);

    cudaMalloc(&dMaxProSupFeature, sizeof(double));
    cudaMalloc(&dMaxProInfFeature, sizeof(double));

    cudaMalloc(&dMeanProSupFeature, sizeof(double));
    cudaMalloc(&dMeanProInfFeature, sizeof(double));

    dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
    dim3 dimGrid (rows/dimBlock.x,cols/dimBlock.y);	

    matInfinityNorm<<<dimGrid,dimBlock>>>(dProSupFeature,dInfNormProSupFeature,
                                          rows, cols, BLOCKSIZE);

    matInfinityNorm<<<dimGrid,dimBlock>>>(dProInfFeature,dInfNormProInfFeature,
                                          rows, cols, BLOCKSIZE);

    //
    // Dividir con scalar
    // ------------------
    divScalarMatrix<<<BlocksInGrid,threadsPerBlock>>>(dProSupFeature,dInfNormProSupFeature, rows*cols);
    divScalarMatrix<<<BlocksInGrid,threadsPerBlock>>>(dProInfFeature,dInfNormProInfFeature, rows*cols);
    
    //
    // Maximum
    // -------
    find_maximum(dProSupFeature, dMaxProSupFeature, rows*cols);
    find_maximum(dProSupFeature, dMaxProInfFeature, rows*cols);

    //
    // Mean
    // ----
    meanMatrix<<<BlocksInGrid,threadsPerBlock>>>(dProSupFeature, dMeanProSupFeature, rows*cols);
    meanMatrix<<<BlocksInGrid,threadsPerBlock>>>(dProInfFeature, dMeanProInfFeature, rows*cols);

    //
    // Apply
    // -----
    applyNormSum<<<BlocksInGrid,threadsPerBlock>>>(dMap,dProSupFeature,dMaxProSupFeature,dMeanProSupFeature,
                                                   dProInfFeature,dMaxProInfFeature,dMeanProInfFeature,
                                                   rows*cols);

    // Liberar memoria
    cudaFree(dInfNormProSupFeature);
    cudaFree(dInfNormProInfFeature);

    cudaFree(dMaxProSupFeature);
    cudaFree(dMaxProInfFeature);

    cudaFree(dMeanProSupFeature);
    cudaFree(dMeanProInfFeature);
}


void getSalency(double* &salency, 
                double* &Imap, double* &Omap, double* &Cmap,
                int rows, int cols) {
	sum3<<<BlocksInGrid,threadsPerBlock>>>(salency,
                                           Imap,Omap,Cmap,
                                           rows*cols);
}


void interpolation(double* &original, double* &result, 
                   int rows, int cols, int factor){
    
    dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
    dim3 dimGrid (rows/dimBlock.x,cols/dimBlock.y);	
    
    kernelInterpolationRow<<<dimGrid,dimBlock>>>(original,result,
                                                 rows,cols,factor);
    kernelInterpolationCol<<<dimGrid,dimBlock>>>(original,result,
                                                 rows,cols,factor);
}

