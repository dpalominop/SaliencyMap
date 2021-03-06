
#ifndef SaliencyMapGPU_H
#define SaliencyMapGPU_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include <stdio.h> 
#include <math.h>
#include <omp.h>

#include "kernel.h"
#include "utils.h"
#include "../Filter/Filter.h"
#include "tools/SaliencyMap/SaliencyMap.h"


#define NUMBER_OF_LEVELS 9
#define THREAD_COUNT     1

using namespace cv;

class SaliencyMapGPU
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

	double **_Salency;

	std::string _dir;

	int rRows, rCols;  // Real size
	int  rows, cols;  // Square size

	int pad1R, pad1C;
	int pad2R, pad2C;

	// Principal functions
	void getPyramids();
	void reductionFeatures();

	void getMap(double** &feature, double** &map, const double kernel[][5]);
	void getSalency();

	void reductionPyramid(double***pyramid, double **reduction);
	void reductionPyramid(double***pyramid, double **reduction, int sup, int inf1, int inf2);
	void centerSurroundDiff(double***pyramid, double** difference, int firstLevel, int secondLevel);
	void centerSurroundDiff(double** &supLevel, double** &infLevel, double ** &difference, int firstLevel, int secondLevel, int endLevel);
	void absDifference(double** out, double** first, double** second, int rows, int cols);

	void imshow(Mat img, std::string name);
	void imshow(double **img,int x_length, int y_length, std::string name);
	
public:
	SaliencyMapGPU(std::string direction): _dir(direction) {
		this->getData();
	}

	void getData();
	void run();
	void showSalency();

	void setDirImage(std::string direction){ _dir = direction; }
};

#endif
