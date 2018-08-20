#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h> 
#include <math.h>
#include "kernel.h"
#include "Filter\Filter.h"

#define NUMBER_OF_LEVELS 9
#define THREAD_COUNT     4

using namespace cv;

double ** allocate(int rows, int cols) {
	double ** dPointer = new double*[rows];
	for (int i = 0; i < rows; ++i) {
		dPointer[i] = new double[cols];
	}
	return dPointer;
}


void cleanMemory(double **dPointer, int rows) {
	for (int i = 0; i < rows; i++)
		delete[] dPointer[i];
	delete[] dPointer;
}


class salencyMap
{
private:
	// Image date
	double ***_R, ***_G,
		   ***_B, ***_Y;
	double ***_I,
		   ***_O0, ***_O45,
		   ***_O90, ***_O135;
	
	int rRows, rCols;  // Real size
	int  rows, cols;  // Square size

	// Principal functions
	void getPyramids();
	void reductionFeatures();

	void reductionPyramid(double***pyramid, double **reduction);
	void reductionPyramid(double***pyramid, double **reduction, int sup, int inf1, int inf2);
	void centerSurroundDiff(double***pyramid, double** difference, int firstLevel, int secondLevel);
	void absDifference(double** out, double** first, double** second, int rows, int cols);

public:
	salencyMap() {
		_I    = new double**[NUMBER_OF_LEVELS];
		_O0   = new double**[NUMBER_OF_LEVELS];
		_O45  = new double**[NUMBER_OF_LEVELS];
		_O90  = new double**[NUMBER_OF_LEVELS];
		_O135 = new double**[NUMBER_OF_LEVELS];

		_R = new double**[NUMBER_OF_LEVELS];
		_G = new double**[NUMBER_OF_LEVELS];
		_B = new double**[NUMBER_OF_LEVELS];
		_Y = new double**[NUMBER_OF_LEVELS];
	}

	void getData();
	void run();
};


void salencyMap::getData() {
/*
	Get image
	---------
 */
	cv::Mat image, padImg;
	image = cv::imread("oso.jpg", CV_LOAD_IMAGE_COLOR);

	if (!image.data){
		std::cout << "Could not open or find the image" << std::endl;
		return ;
	}


/*
	Get size
	--------
 */
	rRows = image.rows;
	rCols = image.cols;

	rows = pow(2.0, ceil(log2((double)rRows)));
	cols = pow(2.0, ceil(log2((double)rCols)));

/*
	Padding
	-------

		     0    pad1C                               pad2C   cols
           0  -------------------------------------------------
		     |                                                 |
	 	     |      0                         startC  rCols    |
	   pad1R |    0  -----------------------------------       |
		     |      |                            .      |      |
		     |      |                            .      |      |
		     |      |                            .      |      |
	 	     |      |                            .      |      |
	 	     |      |                            .      |      |
             |startR| . . . . . . . . . . . . . . . . . |      |
		     |      |                            .      |      |
		     |      |                            .      |      |
	   pad2R | rRows -----------------------------------       |
		     |                                                 |
		     |                                                 |
	    rows  -------------------------------------------------

 */
	padImg = cv::Mat(rows, cols, CV_8UC3);
	int i, j, ip, jp;

	int pad1R = (rows - rRows) / 2;
	int pad1C = (cols - rCols) / 2;

	int pad2R = pad1R + rRows;
	int pad2C = pad1C + rCols;

	int startR = rRows - (rows - pad2R);
	int startC = rCols - (cols - pad2C);

	// Copy image
	for (i = 0, ip = pad1R;i < rRows;++i, ++ip) {
		for (j = 0, jp = pad1C;j < rCols;++j, ++jp) {
			padImg.at<Vec3b>(ip, jp) = image.at<Vec3b>(i, j);
		}
	}

	// Left Padding
	for (i = pad1R;i < pad2R;++i) {
		for (jp = pad1C - 1, j = pad1C + 1; jp > -1; ++j, --jp) {
			padImg.at<Vec3b>(i, jp) = padImg.at<Vec3b>(i, j);
		}
	}

	// Right Padding
	for (i = pad1R;i < pad2R;++i) {
		for (jp = cols - 1, j = startC - 1; jp >= pad2C; ++j, --jp) {
			padImg.at<Vec3b>(i, jp) = padImg.at<Vec3b>(i, j);
		}
	}

	// Higher Padding
	for (ip = pad1R - 1, i = pad1R + 1;ip > -1;++i, --ip) {
		for (j = 0; j < cols; ++j) {
			padImg.at<Vec3b>(ip, j) = padImg.at<Vec3b>(i, j);
		}
	}

	// Lower Padding
	for (ip = rows - 1, i = startR - 1; ip >= pad2R; ++i, --ip) {
		for (j = 0; j < cols; ++j) {
			padImg.at<Vec3b>(ip, j) = padImg.at<Vec3b>(i, j);
		}
	}


/*
	Inicialize
	----------
 */
	for (int k = 0; k < NUMBER_OF_LEVELS;++k){
		_I  [k] = allocate(rows,cols);
		_O0 [k] = allocate(rows, cols); _O45 [k] = allocate(rows, cols);
		_O90[k] = allocate(rows, cols); _O135[k] = allocate(rows, cols);
		_R  [k] = allocate(rows, cols); _G   [k] = allocate(rows, cols);
		_B  [k] = allocate(rows, cols); _Y   [k] = allocate(rows, cols);
	}

/*
	Get features
	------------
 */
	uint8_t* pixelPtr = (uint8_t*)padImg.data;
	int cn = padImg.channels();
	double r, g, b;
	double aux;

#   pragma omp parallel for collapse(2) num_threads(THREAD_COUNT)
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			// Get values
			b = (double)pixelPtr[i*cols*cn + j * cn + 0]; // B
			g = (double)pixelPtr[i*cols*cn + j * cn + 1]; // G
			r = (double)pixelPtr[i*cols*cn + j * cn + 2]; // R
				
			aux = (r + g + b) / 3.0;
			_I   [0][i][j] = aux;
			_O0  [0][i][j] = aux;
			_O45 [0][i][j] = aux;
			_O90 [0][i][j] = aux;
			_O135[0][i][j] = aux;
			
			aux = r - (g + b) / 2.0;
			_R[0][i][j] = (aux > 0.0) ? aux : 0.0;

			aux = g - (b + r) / 2.0;
			_G[0][i][j] = (aux > 0.0) ? aux : 0.0;

			aux = b - (r + g) / 2.0;
			_B[0][i][j] = (aux > 0.0) ? aux : 0.0;

			aux = (r + g) / 2.0 - abs(r - g) / 2.0 - b;
			_Y[0][i][j] = (aux > 0.0) ? aux : 0.0;

		}
	}

}

void salencyMap::run(){
	this->getPyramids();
	this->reductionFeatures();
}

void salencyMap::getPyramids() {
	
	Filter gauss    = Filter(GAUSS_KERNEL);
	Filter gabor0   = Filter(GABOR_00_KERNEL);
	Filter gabor45  = Filter(GABOR_45_KERNEL);
	Filter gabor90  = Filter(GABOR_90_KERNEL);
	Filter gabor135 = Filter(GABOR_135_KERNEL);

	for (int k = 1; k < NUMBER_OF_LEVELS;++k) {
		gauss   .convolution(_I   [k - 1], rows, _I   [k], THREAD_COUNT, 2);
		gabor0  .convolution(_O0  [k - 1], rows, _O0  [k], THREAD_COUNT, 2);
		gabor45 .convolution(_O45 [k - 1], rows, _O45 [k], THREAD_COUNT, 2);
		gabor90 .convolution(_O90 [k - 1], rows, _O90 [k], THREAD_COUNT, 2);
		gabor135.convolution(_O135[k - 1], rows, _O135[k], THREAD_COUNT, 2);

		gauss.convolution(_R[k - 1], rows, _R[k], THREAD_COUNT, 2);
		gauss.convolution(_G[k - 1], rows, _G[k], THREAD_COUNT, 2);
		gauss.convolution(_B[k - 1], rows, _B[k], THREAD_COUNT, 2);
		gauss.convolution(_Y[k - 1], rows, _Y[k], THREAD_COUNT, 2);
	}
	
}

void salencyMap::reductionPyramid(double***pyramid, double **reduction) {
	reductionPyramid(pyramid, reduction, 2, 5, 6);
	reductionPyramid(pyramid, reduction, 3, 6, 7);
	reductionPyramid(pyramid, reduction, 4, 7, 8);
}


void salencyMap::reductionPyramid(double***pyramid, double **reduction, int sup, int inf1, int inf2) {
	double **imSI1, **imSI2;
	double _norm, _max, _mean;
	double coeff1, coeff2;

	imSI1 = allocate(rows, cols);
	imSI2 = allocate(rows, cols);

	centerSurroundDiff(pyramid, imSI1, sup, inf1);
	centerSurroundDiff(pyramid, imSI2, sup, inf2);
	/*
	_norm = norm(imSI1, rows, cols);
	_max  =  max(imSI1, rows, cols);
	_mean = mean(imSI1, rows, cols);
	coeff1 = (_max - _mean)*(_max - _mean) / _norm;

	_norm = norm(imSI2, rows, cols);
	_max  =  max(imSI2, rows, cols);
	_mean = mean(imSI2, rows, cols);
	coeff2 = (_max - _mean)*(_max - _mean) / _norm;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < rows; ++j) {
			reduction[i][j] += coeff1 * imSI1[i][j];
			reduction[i][j] += coeff2 * imSI2[i][j];
		}
	}
	*/

	cleanMemory(imSI1, rows);
	cleanMemory(imSI2, rows);
}


void salencyMap::reductionFeatures() {
	double **I, 
		   **R , **G  , **B  , **Y,
		   **O0, **O45, **O90, **O135;

	I = allocate(rows, cols);
	R = allocate(rows, cols); G = allocate(rows, cols);
	B = allocate(rows, cols); Y = allocate(rows, cols);
	O0  = allocate(rows, cols); O45  = allocate(rows, cols);
	O90 = allocate(rows, cols); O135 = allocate(rows, cols);

#   pragma omp parallel for collapse(2) num_threads(THREAD_COUNT)
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < rows; ++j) {
			I  [i][j] = 0;
			R  [i][j] = 0; G   [i][j] = 0;
			B  [i][j] = 0; Y   [i][j] = 0;
			O0 [i][j] = 0; O45 [i][j] = 0;
			O90[i][j] = 0; O135[i][j] = 0;
		}
	}
	
	reductionPyramid(_I  ,  I );
	reductionPyramid(_R  ,  R ); reductionPyramid(_G   , G   );
	reductionPyramid(_B  ,  B ); reductionPyramid(_Y   , Y   );
	reductionPyramid(_O0 , O0 ); reductionPyramid(_O45 , O45 );
	reductionPyramid(_O90, O90); reductionPyramid(_O135, O135);
}

void salencyMap::centerSurroundDiff(double***pyramid, double** difference,int firstLevel, int secondLevel) {
	double **Ifs;
	Ifs = allocate(rows, cols);
	//Ifs = interpolation(_I[firstLevel], pow(2, secondLevel- firstLevel) );
	absDifference(difference, _I[firstLevel],Ifs,rows,cols);

	//difference = interpolation(difference, pow(2,firstLevel));
	//difference = interpolation(difference, pow(2,firstLevel));
}

void salencyMap::absDifference(double** out, double** first, double** second, int rows, int cols) {
	double a, b;

#   pragma omp parallel for collapse(2) num_threads(THREAD_COUNT)
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			a =  first[i][j];
			b = second[i][j];
			out[i][j] = (a > b) ? (a - b) : (b - a);
		}
	}
}
