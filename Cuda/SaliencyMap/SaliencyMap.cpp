#include "SaliencyMap.h"

#include <iostream>

void SaliencyMap::getData() {
	/*
		Get image
		---------
	 */
	cv::Mat image, padImg;
	image = cv::imread(_dir, CV_LOAD_IMAGE_COLOR);

	if (!image.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return;
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

	pad1R = (rows - rRows) / 2;
	pad1C = (cols - rCols) / 2;

	pad2R = pad1R + rRows;
	pad2C = pad1C + rCols;

	int startR = pad2R - (rows - pad2R);
	int startC = pad2C - (cols - pad2C);

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

	//SaliencyMap::imshow(padImg);

/*
	Inicialize
	----------
 */
	gpuHostAlloc(_I  ,rows, cols);
	gpuHostAlloc(_O0 ,rows, cols); gpuHostAlloc(_O45 ,rows, cols);
	gpuHostAlloc(_O90,rows, cols); gpuHostAlloc(_O135,rows, cols);
	gpuHostAlloc(_R  ,rows, cols); gpuHostAlloc(_G   ,rows, cols);
	gpuHostAlloc(_B  ,rows, cols); gpuHostAlloc(_Y   ,rows, cols);

	gpuMalloc(_Imap,rows/4, cols/4);	// To -> 0.0
	gpuMalloc(_Cmap,rows/4, cols/4);	// To -> 0.0
	gpuMalloc(_Omap,rows/4, cols/4);	// To -> 0.0

	gpuHostAlloc(_Salency,rows/4, cols/4);

	/*
		Get features
		------------
	 */
	double r, g, b;
	double aux;

	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			Vec3b bgrPixel = padImg.at<Vec3b>(i, j);

			// Get values
			b = (double)bgrPixel[0]; // B
			g = (double)bgrPixel[1]; // G
			r = (double)bgrPixel[2]; // R

			aux = (r + g + b) / 3.0;
			_I   [ i*cols + j ] = aux;	//(j)*(id) ) + (i)
			_O0  [ i*cols + j ] = aux;
			_O45 [ i*cols + j ] = aux;
			_O90 [ i*cols + j ] = aux;
			_O135[ i*cols + j ] = aux;

			aux = r - (g + b) / 2.0;
			_R[ i*cols + j ] = (aux > 0.0) ? aux : 0.0;

			aux = g - (b + r) / 2.0;
			_G[ i*cols + j ] = (aux > 0.0) ? aux : 0.0;

			aux = b - (r + g) / 2.0;
			_B[ i*cols + j ] = (aux > 0.0) ? aux : 0.0;

			aux = (r + g) / 2.0 - abs(r - g) / 2.0 - b;
			_Y[ i*cols + j ] = (aux > 0.0) ? aux : 0.0;
		}
	}
	/*
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			std::cout << _I[ j*rows + i ] << "\t";
		}
		std::cout << std::endl;
	}
	*/
}


void SaliencyMap::run() {
	getMap(_I,    _Imap, GAUSS_KERNEL    ,rows,cols);
	getMap(_O0,   _Omap, GABOR_00_KERNEL ,rows,cols);
	getMap(_O45,  _Omap, GABOR_45_KERNEL ,rows,cols);
	getMap(_O90,  _Omap, GABOR_90_KERNEL ,rows,cols);
	getMap(_O135, _Omap, GABOR_135_KERNEL,rows,cols);
	getMap(_R,    _Cmap, GAUSS_KERNEL    ,rows,cols);
	getMap(_G,    _Cmap, GAUSS_KERNEL    ,rows,cols);
	getMap(_B,    _Cmap, GAUSS_KERNEL    ,rows,cols);
	getMap(_Y,    _Cmap, GAUSS_KERNEL    ,rows,cols);

	gpuImshow(_Imap, rows/4, cols/4);

	getSalency(_Salency, _Imap,_Omap,_Cmap,rows/4,cols/4);

	gpuFreeHostAlloc(_I  );
	gpuFreeHostAlloc(_R  ); gpuFreeHostAlloc(_G   ); 
	gpuFreeHostAlloc(_B  ); gpuFreeHostAlloc(_Y   );
	gpuFreeHostAlloc(_O0 ); gpuFreeHostAlloc(_O45 ); 
	gpuFreeHostAlloc(_O90); gpuFreeHostAlloc(_O135);

	gpuFreeMalloc(_Imap); gpuFreeMalloc(_Omap); gpuFreeMalloc(_Cmap);
}


void SaliencyMap::showSalency(){
	int local_rows = (pad2R/4)-(pad1R/4);
	int local_cols = (pad2C/4)-(pad1C/4);

	double maxImg, minImg, coeff;

	maxArray(_Salency, maxImg, local_rows*local_cols, THREAD_COUNT);
	minArray(_Salency, minImg, local_rows*local_cols, THREAD_COUNT);

	std::cout << "En Saliency:" << std::endl;
	std::cout << "min: " << minImg << ", max: " << maxImg << std::endl;

/*
	for (int i = 0; i < local_rows; i++) {
		for (int j = 0; j < local_cols; j++) {
			std::cout << _Salency[ j*rows/4 + i ] << "\t";
		}
		std::cout << std::endl;
	}
*/

	coeff = 255 / (maxImg - minImg);

	Mat out = cv::Mat(local_rows, local_cols, CV_8UC1);

	for (int i = 0; i < local_rows; i++) {
		for (int j = 0; j < local_cols; j++) {
			out.at<uchar>(i, j) = (uchar)(coeff*(_Salency[ j*rows/4 + i ] - minImg));
		}
	}

	resize(out, out, Size(), 2, 2, INTER_LANCZOS4);

	SaliencyMap::imshow(out, "Salency Map!!");
}



void SaliencyMap::imshow(double **img, int x_length, int y_length, std::string name = "Una ventana") {
	double maxImg, minImg, coeff;

	maxArray(img, maxImg, x_length, y_length, THREAD_COUNT);
	minArray(img, minImg, x_length, y_length, THREAD_COUNT);

	coeff = 255 / (maxImg - minImg);

	Mat out = cv::Mat(x_length, y_length, CV_8UC1);

	for (int i = 0; i < x_length; i++) {
		for (int j = 0; j < y_length; j++) {
			out.at<uchar>(i, j) = (uchar)(coeff*(img[i][j] - minImg));
		}
	}
	
	SaliencyMap::imshow(out, name);
}


void SaliencyMap::imshow(Mat img, std::string name = "Una ventana") {
	cv::imshow(name, img);
	waitKey(0);
}

