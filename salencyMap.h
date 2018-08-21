
#ifndef SALENCYMAP_H
#define SALENCYMAP_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h> 
#include <math.h>
#include <string>
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

int pow2(int pot){
	int out = 1;
	for (int i=0; i<pot; ++i) out *= 2;
	return out;
}


struct Pyramid{
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

	Pyramid(int _rows, int _cols): rows(_rows), cols(_cols) {
		int c;

		_Base = new double*[rows];
		for(int i=0; i<rows; ++i) _Base[i] = new double[rows];

		c = 2; _Level1 = new double*[rows/c];
		for(int i=0; i<rows/c; ++i) _Level1[i] = new double[rows/c];

		c = 4; _Level2 = new double*[rows/c];
		for(int i=0; i<rows/c; ++i) _Level2[i] = new double[rows/c];

		c = 8; _Level3 = new double*[rows/c];
		for(int i=0; i<rows/c; ++i) _Level3[i] = new double[rows/c];

		c = 16; _Level4 = new double*[rows/c];
		for(int i=0; i<rows/c; ++i) _Level4[i] = new double[rows/c];

		c = 32; _Level5 = new double*[rows/c];
		for(int i=0; i<rows/c; ++i) _Level5[i] = new double[rows/c];

		c = 64; _Level6 = new double*[rows/c];
		for(int i=0; i<rows/c; ++i) _Level6[i] = new double[rows/c];

		c = 128; _Level7 = new double*[rows/c];
		for(int i=0; i<rows/c; ++i) _Level7[i] = new double[rows/c];

		c = 256; _Level8 = new double*[rows/c];
		for(int i=0; i<rows/c; ++i) _Level8[i] = new double[rows/c];
	}

	void clean(){
		int c;
		
		for(int i=0; i<rows; ++i) delete [] _Base[i];
		delete [] _Base;

		c = 2; for(int i=0; i<rows/c; ++i) delete [] _Level1[i];
		delete [] _Level1;

		c = 4; for(int i=0; i<rows/c; ++i) delete [] _Level2[i];
		delete [] _Level2;

		c = 8; for(int i=0; i<rows/c; ++i) delete [] _Level3[i];
		delete [] _Level3;

		c = 16; for(int i=0; i<rows/c; ++i) delete [] _Level4[i];
		delete [] _Level4;

		c = 32; for(int i=0; i<rows/c; ++i) delete [] _Level5[i];
		delete [] _Level5;

		c = 64; for(int i=0; i<rows/c; ++i) delete [] _Level6[i];
		delete [] _Level6;

		c = 128; for(int i=0; i<rows/c; ++i) delete [] _Level7[i];
		delete [] _Level7;

		c = 256; for(int i=0; i<rows/c; ++i) delete [] _Level8[i];
		delete [] _Level8;

	}
};




class salencyMap
{
private:
	// Image date
	double **_R, **_G,
		   **_B, **_Y;
	double **_I,
		   **_O0, **_O45,
		   **_O90, **_O135;

	double **_Imap, **_Omap,
	       **_Cmap;

	std::string _dir;

	int rRows, rCols;  // Real size
	int  rows, cols;  // Square size

	int startRealR, startRealC;

	// Principal functions
	void getPyramids();
	void reductionFeatures();

	void getMap(double** &feature, double** &map, double kernel[][5]);
	void getSalency();

	void reductionPyramid(double***pyramid, double **reduction);
	void reductionPyramid(double***pyramid, double **reduction, int sup, int inf1, int inf2);
	void centerSurroundDiff(double***pyramid, double** difference, int firstLevel, int secondLevel);
	void centerSurroundDiff(double** &supLevel, double** &infLevel, double ** &difference, int firstLevel, int secondLevel, int endLevel);
	void absDifference(double** out, double** first, double** second, int rows, int cols);

	void imshow(Mat img, std::string name);
	void imshow(double **img,int x_length, int y_length, std::string name);

public:
	salencyMap(std::string direction): _dir(direction) {
		this->getData();
	}

	void getData();
	void run();

	void setDirImage(std::string direction){ _dir = direction; }
};


void salencyMap::getData() {
/*
	Get image
	---------
 */
	cv::Mat image, padImg;
	image = cv::imread(_dir, CV_LOAD_IMAGE_COLOR);

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
	_I  = allocate(rows, cols);
	_O0 = allocate(rows, cols); _O45 = allocate(rows, cols);
	_O90= allocate(rows, cols); _O135= allocate(rows, cols);
	_R  = allocate(rows, cols); _G   = allocate(rows, cols);
	_B  = allocate(rows, cols); _Y   = allocate(rows, cols);
	
	_Imap = allocate(rows/4, cols/4);
	_Cmap = allocate(rows/4, cols/4);
	_Omap = allocate(rows/4, cols/4);


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
			_I   [i][j] = aux;
			_O0  [i][j] = aux;
			_O45 [i][j] = aux;
			_O90 [i][j] = aux;
			_O135[i][j] = aux;
			
			aux = r - (g + b) / 2.0;
			_R[i][j] = (aux > 0.0) ? aux : 0.0;

			aux = g - (b + r) / 2.0;
			_G[i][j] = (aux > 0.0) ? aux : 0.0;

			aux = b - (r + g) / 2.0;
			_B[i][j] = (aux > 0.0) ? aux : 0.0;

			aux = (r + g) / 2.0 - abs(r - g) / 2.0 - b;
			_Y[i][j] = (aux > 0.0) ? aux : 0.0;

			_Imap[i][j] = 0;
			_Omap[i][j] = 0;
			_Cmap[i][j] = 0;
		}
	}
}

void salencyMap::run(){
	this->getMap(_I   ,_Imap,GAUSS_KERNEL   );
	this->getMap(_O0  ,_Omap,GABOR_00_KERNEL);
	this->getMap(_O45 ,_Omap,GABOR_45_KERNEL);
	this->getMap(_O90 ,_Omap,GABOR_90_KERNEL);
	this->getMap(_O135,_Omap,GABOR_135_KERNEL);
	this->getMap(_R   ,_Cmap,GAUSS_KERNEL    );
	this->getMap(_G   ,_Cmap,GAUSS_KERNEL    );
	this->getMap(_B   ,_Cmap,GAUSS_KERNEL    );
	this->getMap(_Y   ,_Cmap,GAUSS_KERNEL    );

	// Print images
	imshow(_Imap,rows/4,cols/4,"Mapa de Intensidad");
	imshow(_Omap,rows/4,cols/4,"Mapa de Orientacion");	
	imshow(_Cmap,rows/4,cols/4,"Mapa de Color");

}

void salencyMap::getMap(double** &feature, double** &map, double kernel[][5]){
	Pyramid py(rows,cols);
	Filter blur(kernel);

	// Generate pyramid
	blur.convolution(   feature, rows,cols, py._Level1, 2, THREAD_COUNT);
	blur.convolution(py._Level1, rows,cols, py._Level2, 2, THREAD_COUNT);
	blur.convolution(py._Level2, rows,cols, py._Level3, 2, THREAD_COUNT);
	blur.convolution(py._Level3, rows,cols, py._Level4, 2, THREAD_COUNT);
	blur.convolution(py._Level4, rows,cols, py._Level5, 2, THREAD_COUNT);
	blur.convolution(py._Level5, rows,cols, py._Level6, 2, THREAD_COUNT);
	blur.convolution(py._Level6, rows,cols, py._Level7, 2, THREAD_COUNT);
	blur.convolution(py._Level7, rows,cols, py._Level8, 2, THREAD_COUNT);
	
	// Center-surround difference
	double **feat25 = allocate(rows/4, cols/4), **feat26 = allocate(rows/4, cols/4);
	double **feat36 = allocate(rows/4, cols/4), **feat37 = allocate(rows/4, cols/4);
	double **feat47 = allocate(rows/4, cols/4), **feat48 = allocate(rows/4, cols/4);

	centerSurroundDiff(py._Level2, py._Level5, feat25, 2, 5, 2);
	centerSurroundDiff(py._Level2, py._Level6, feat26, 2, 6, 2);

	centerSurroundDiff(py._Level3, py._Level6, feat36, 3, 6, 2);
	centerSurroundDiff(py._Level3, py._Level7, feat37, 3, 7, 2);

	centerSurroundDiff(py._Level4, py._Level7, feat47, 4, 7, 2);
	centerSurroundDiff(py._Level4, py._Level8, feat48, 4, 8, 2);

	// Clean Pyramid
	py.clean();

	// Normalizarion
	nrm(feat25,rows,cols,THREAD_COUNT);
	nrm(feat26,rows,cols,THREAD_COUNT);
	
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			map[i][j] += feat25[i][j] + feat26[i][j];
		}
	}

	nrm(feat36,rows,cols,THREAD_COUNT);
	nrm(feat37,rows,cols,THREAD_COUNT);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			map[i][j] += feat36[i][j] + feat37[i][j];
		}
	}

	nrm(feat47,rows,cols,THREAD_COUNT);
	nrm(feat48,rows,cols,THREAD_COUNT);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			map[i][j] += feat47[i][j] + feat48[i][j];
		}
	}
}


void salencyMap::centerSurroundDiff(double** &supLevel, double** &lowLevel, double ** &difference, int sup, int low, int endl){
	int supRow = rows/pow2(sup);
	int supCol = cols/pow2(sup);

	int lowRow = rows/pow2(low);
	int lowCol = cols/pow2(low);

	double **growLowLevel = allocate(supRow,supCol);
	Filter::growthMatrix(lowLevel,lowRow, lowCol,growLowLevel,pow2(sup-low),THREAD_COUNT);

	if(sup != endl){
		double **rawDifference = allocate(supRow, supCol);

		absDifference(rawDifference, supLevel,growLowLevel,supRow,supCol);
		Filter::growthMatrix(rawDifference,supRow,supCol,difference,pow2(endl-sup),THREAD_COUNT);
	}else{
		absDifference(difference, supLevel,growLowLevel,supRow,supCol);
	}
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


void salencyMap::imshow(double **img,int x_length, int y_length, std::string name = "Una ventana"){
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

	salencyMap::imshow(out,name);
}


void salencyMap::imshow(Mat img, std::string name = "Una ventana"){
	//Mat half( img.rows/2, img.cols/2,  img.type() );
	//cv::resize(img,half,cv::Size(), 0.5, 0.5);
	cv::imshow(name,img);
	waitKey(0);
}

#endif