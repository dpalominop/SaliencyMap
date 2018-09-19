
#ifndef SALIENCYMAP_H
#define SALIENCYMAP_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include <stdio.h> 
#include <math.h>
#include <omp.h>

#include "cublas_v2.h"
#include "kernel.h"
#include "utils.h"
#include "../../OpenMP/Filter/Filter.h"

#define NUMBER_OF_LEVELS 9
#define THREAD_COUNT     1

#define GRIDSIZE 10
#define BLOCKSIZE 32

#define threadsPerBlock 512
#define BlocksInGrid    20

using namespace cv;

class SaliencyMap
{
private:
	// Image date
	double *_R, *_G,
		   *_B, *_Y;
	double *_I,
		   *_O0,  *_O45,
		   *_O90, *_O135;

	double *_Imap, *_Omap,
	       *_Cmap;

	double *_Salency;

	std::string _dir;

	int rRows, rCols;  // Real size
	int  rows, cols;  // Square size

	int pad1R, pad1C;
	int pad2R, pad2C;

	// Principal functions
	void getPyramids();
	void reductionFeatures();

	//void getMap(double* &feature, double* &map, const double kernel[][5]);
	//void getSalency();

	//void centerSurroundDiff(double** &supLevel, double** &infLevel, double ** &difference, int firstLevel, int secondLevel, int endLevel);
	//void absDifference(double** out, double** first, double** second, int rows, int cols);

	void imshow(Mat img, std::string name);
	void imshow(double **img,int x_length, int y_length, std::string name);
	
public:
	SaliencyMap(std::string direction): _dir(direction) {
		this->getData();
	}

	void getData();
	void run();
	void showSalency();

	void setDirImage(std::string direction){ _dir = direction; }
};

////////////////////////////////////////////////////////////////////////////////
// GPU functions
////////////////////////////////////////////////////////////////////////////////
void getMap(double* &feature, double* &map, 
                       const double kernel[][5],
					   int rows, int cols);
void nrmSumGPU(double* &dProSupFeature, double* &dProInfFeature, 
			   double* &dMap,
			   int rows, int cols);
void centerSurroundDiffGPU(double* &dSupLevel, double* &dLowLevel,
                                      double* &dDifference, 
									  int sup, int low, int endl,
									  int rows, int cols);

void getSalency(double* &salency, 
                double* &Imap, double* &Omap, double* &Cmap,
                int rows, int cols);

void gpuHostAlloc(double*& d_p, int rows, int cols);
void gpuMalloc(double*& d_p, int rows, int cols);

void gpuFreeHostAlloc(double*& d_p);
void gpuFreeMalloc(double*& d_p);

#endif