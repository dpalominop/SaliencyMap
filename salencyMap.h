
#ifndef SALENCYMAP_H
#define SALENCYMAP_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h> 
#include <math.h>
#include "kernel.h"
#include "Filter/Filter.h"
#include "utils.h"

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

	int startRealR, startRealC;

	// Principal functions
	void getPyramids();
	void reductionFeatures();

	void reductionPyramid(double***pyramid, double **reduction);
	void reductionPyramid(double***pyramid, double **reduction, int sup, int inf1, int inf2);
	void centerSurroundDiff(double***pyramid, double** difference, int firstLevel, int secondLevel);
	void absDifference(double** out, double** first, double** second, int rows, int cols);

	void imshow(Mat img);
	void imshow(double **img,int x_length, int y_length);

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


void salencyMap::imshow(double **img,int x_length, int y_length){
	double maxImg, minImg, coeff;

	maxArray(img,maxImg,x_length,y_length,THREAD_COUNT);
	minArray(img,minImg,x_length,y_length,THREAD_COUNT);

	coeff = 255/(maxImg-minImg);

	Mat out = cv::Mat(x_length, y_length, CV_8UC1);

	for (int i = 0; i < x_length; i++) {
		for (int j = 0; j < y_length; j++) {
			out.at<uchar>(i, j) = (uchar)( coeff*(img[i][j]-minImg) );
		}
	}

	salencyMap::imshow(out);
}


void salencyMap::imshow(Mat img){
	Mat half( img.rows/2, img.cols/2,  img.type() );
	cv::resize(img,half,cv::Size(), 0.5, 0.5);
	cv::imshow("No quiero jalar!",half);
	waitKey(0);
}

void salencyMap::getData() {
/*
	Get image
	---------
 */
	cv::Mat image, padImg;
	image = cv::imread("images/oso.jpg", CV_LOAD_IMAGE_COLOR);

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

	int startR = pad2R - (rows - pad2R);
	int startC = pad2C - (cols - pad2C);

	startRealR = pad1R;
	startRealC = pad1C;

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
		for (jp = cols - 1, j = startC; jp >= pad2C; ++j, --jp) {
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
	for (ip = rows - 1, i = startR; ip >= pad2R; ++i, --ip) {
		for (j = 0; j < cols; ++j) {
			padImg.at<Vec3b>(ip, j) = padImg.at<Vec3b>(i, j);
		}
	}

	//salencyMap::imshow(padImg);

/*
	Inicialize
	----------
 */
	for (int k = 0; k < NUMBER_OF_LEVELS;++k){
		_I  [k] = allocate(rows, cols);
		_O0 [k] = allocate(rows, cols); _O45 [k] = allocate(rows, cols);
		_O90[k] = allocate(rows, cols); _O135[k] = allocate(rows, cols);
		_R  [k] = allocate(rows, cols); _G   [k] = allocate(rows, cols);
		_B  [k] = allocate(rows, cols); _Y   [k] = allocate(rows, cols);
	}

/*
	Get features
	------------
 */
	double r, g, b;
	double aux;

//#   pragma omp parallel for collapse(2) num_threads(THREAD_COUNT)
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			Vec3b bgrPixel = padImg.at<Vec3b>(i, j);

			// Get values
			b = (double)bgrPixel[0]; // B
			g = (double)bgrPixel[1]; // G
			r = (double)bgrPixel[2]; // R
			
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
		gauss   .convolution(_I   [k - 1], rows, cols, _I   [k], 2,THREAD_COUNT);
		gabor0  .convolution(_O0  [k - 1], rows, cols, _O0  [k], 2,THREAD_COUNT);
		gabor45 .convolution(_O45 [k - 1], rows, cols, _O45 [k], 2,THREAD_COUNT);
		gabor90 .convolution(_O90 [k - 1], rows, cols, _O90 [k], 2,THREAD_COUNT);
		gabor135.convolution(_O135[k - 1], rows, cols, _O135[k], 2,THREAD_COUNT);

		gauss.convolution(_R[k - 1], rows, cols, _R[k], 2,THREAD_COUNT);
		gauss.convolution(_G[k - 1], rows, cols, _G[k], 2,THREAD_COUNT);
		gauss.convolution(_B[k - 1], rows, cols, _B[k], 2,THREAD_COUNT);
		gauss.convolution(_Y[k - 1], rows, cols, _Y[k], 2,THREAD_COUNT);
	}
	
	imshow(_I[1],rows,cols);
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
	reductionPyramid(_O0 , O0 ); reductionPyramid(_O45 , O45 );
	reductionPyramid(_O90, O90); reductionPyramid(_O135, O135);
	reductionPyramid(_R  ,  R ); reductionPyramid(_G   , G   );
	reductionPyramid(_B  ,  B ); reductionPyramid(_Y   , Y   );

	double **salency;
	salency = allocate(rRows, rCols);

	for (int i = 0, ip = startRealR; i < rRows; ++i, ++ip) {
		for (int j = 0, jp = startRealC; j < rCols; ++j, ++jp) {
			salency[i][j] = I[ip][jp] + O0[ip][jp] + O90[ip][jp] + O45[ip][jp] + O135[ip][jp] + R[ip][jp] + G[ip][jp] + B[ip][jp]  + Y[ip][jp];
		}
	}

	//salencyMap::imshow(salency,rRows, rCols);
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
	
	norm2Array(imSI1, _norm, rows, cols, THREAD_COUNT);
	  maxArray(imSI1, _max , rows, cols, THREAD_COUNT);
	 meanArray(imSI1, _mean, rows, cols, THREAD_COUNT);
	coeff1 = (_max - _mean)*(_max - _mean) / _norm;

	norm2Array(imSI2, _norm, rows, cols, THREAD_COUNT);
	  maxArray(imSI2, _max , rows, cols, THREAD_COUNT);
	 meanArray(imSI2, _mean, rows, cols, THREAD_COUNT);
	coeff2 = (_max - _mean)*(_max - _mean) / _norm;


	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			reduction[i][j] += coeff1 * imSI1[i][j];
			reduction[i][j] += coeff2 * imSI2[i][j];
		}
	}
	
	Filter::deleteMemory(imSI1, rows,cols);
	Filter::deleteMemory(imSI2, rows,cols);
}

void salencyMap::centerSurroundDiff(double***pyramid, double** difference,int firstLevel, int secondLevel) {
	double **Ifs;
	double **rawDiff;
	Ifs = allocate(rows, cols);
	rawDiff = allocate(rows, cols);
	//interpolation(_I[firstLevel], rows, cols, Ifs, pow(2, secondLevel- firstLevel), THREAD_COUNT);
	absDifference(difference, _I[firstLevel],Ifs,rows,cols);

	//interpolation(rawDiff, rows, cols, difference, pow(2, firstLevel), THREAD_COUNT);

	Filter::deleteMemory(    Ifs,rows,cols);
	Filter::deleteMemory(rawDiff,rows,cols);
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

#endif