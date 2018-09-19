#include "SaliencyMap.h"

void SaliencyMapGPU::getData() {
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

	//SaliencyMapGPU::imshow(padImg);

/*
	Inicialize
	----------
 */
	_I = allocate(rows, cols);
	_O0 = allocate(rows, cols); _O45 = allocate(rows, cols);
	_O90 = allocate(rows, cols); _O135 = allocate(rows, cols);
	_R = allocate(rows, cols); _G = allocate(rows, cols);
	_B = allocate(rows, cols); _Y = allocate(rows, cols);

	_Imap = allocate(rows / 4, cols / 4);
	_Cmap = allocate(rows / 4, cols / 4);
	_Omap = allocate(rows / 4, cols / 4);


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
			_I[i][j] = aux;
			_O0[i][j] = aux;
			_O45[i][j] = aux;
			_O90[i][j] = aux;
			_O135[i][j] = aux;

			aux = r - (g + b) / 2.0;
			_R[i][j] = (aux > 0.0) ? aux : 0.0;

			aux = g - (b + r) / 2.0;
			_G[i][j] = (aux > 0.0) ? aux : 0.0;

			aux = b - (r + g) / 2.0;
			_B[i][j] = (aux > 0.0) ? aux : 0.0;

			aux = (r + g) / 2.0 - abs(r - g) / 2.0 - b;
			_Y[i][j] = (aux > 0.0) ? aux : 0.0;
		}
	}

	for (i = 0; i < rows / 4; i++) {
		for (j = 0; j < cols / 4; j++) {
			_Imap[i][j] = 0;
			_Omap[i][j] = 0;
			_Cmap[i][j] = 0;
		}
	}

}

void SaliencyMapGPU::run() {
	this->getMap(_I, _Imap, GAUSS_KERNEL_GPU);
	this->getMap(_O0, _Omap, GABOR_00_KERNEL_GPU);
	this->getMap(_O45, _Omap, GABOR_45_KERNEL_GPU);
	this->getMap(_O90, _Omap, GABOR_90_KERNEL_GPU);
	this->getMap(_O135, _Omap, GABOR_135_KERNEL_GPU);
	this->getMap(_R, _Cmap, GAUSS_KERNEL_GPU);
	this->getMap(_G, _Cmap, GAUSS_KERNEL_GPU);
	this->getMap(_B, _Cmap, GAUSS_KERNEL_GPU);
	this->getMap(_Y, _Cmap, GAUSS_KERNEL_GPU);

	// Print images
	//SaliencyMapGPU::imshow(_Imap, rows / 4, cols / 4, "Mapa de Intensidad");
	//SaliencyMapGPU::imshow(_Omap, rows / 4, cols / 4, "Mapa de Orientacion");
	//SaliencyMapGPU::imshow(_Cmap, rows / 4, cols / 4, "Mapa de Color");

	this->getSalency();
}

void SaliencyMapGPU::getMap(double** &feature, double** &map, const double kernel[][5]) {
	Pyramid py(rows, cols);
	FilterGPU blur(kernel);

	// Generate pyramid
	blur.convolution(feature, rows, cols, py._Level1, 2, THREAD_COUNT);
	blur.convolution(py._Level1, rows / 2, cols / 2, py._Level2, 2, THREAD_COUNT);
	blur.convolution(py._Level2, rows / 4, cols / 4, py._Level3, 2, THREAD_COUNT);
	blur.convolution(py._Level3, rows / 8, cols / 8, py._Level4, 2, THREAD_COUNT);
	blur.convolution(py._Level4, rows / 16, cols / 16, py._Level5, 2, THREAD_COUNT);
	blur.convolution(py._Level5, rows / 32, cols / 32, py._Level6, 2, THREAD_COUNT);
	blur.convolution(py._Level6, rows / 64, cols / 64, py._Level7, 2, THREAD_COUNT);
	blur.convolution(py._Level7, rows / 128, cols / 128, py._Level8, 2, THREAD_COUNT);

	// Center-surround difference
	double **feat25 = allocate(rows / 4, cols / 4), **feat26 = allocate(rows / 4, cols / 4);
	double **feat36 = allocate(rows / 4, cols / 4), **feat37 = allocate(rows / 4, cols / 4);
	double **feat47 = allocate(rows / 4, cols / 4), **feat48 = allocate(rows / 4, cols / 4);

	centerSurroundDiff(py._Level2, py._Level5, feat25, 2, 5, 2);
	centerSurroundDiff(py._Level2, py._Level6, feat26, 2, 6, 2);

	centerSurroundDiff(py._Level3, py._Level6, feat36, 3, 6, 2);
	centerSurroundDiff(py._Level3, py._Level7, feat37, 3, 7, 2);

	centerSurroundDiff(py._Level4, py._Level7, feat47, 4, 7, 2);
	centerSurroundDiff(py._Level4, py._Level8, feat48, 4, 8, 2);

	// Clean Pyramid
	py.clean();

	// Normalizarion
	nrm(feat25, rows / 4, cols / 4, THREAD_COUNT);
	nrm(feat26, rows / 4, cols / 4, THREAD_COUNT);

#pragma omp parallel for collapse(2) num_threads(THREAD_COUNT)
	for (int i = 0; i < rows / 4; i++) {
		for (int j = 0; j < cols / 4; j++) {
			map[i][j] += feat25[i][j] + feat26[i][j];
		}
	}

	nrm(feat36, rows / 4, cols / 4, THREAD_COUNT);
	nrm(feat37, rows / 4, cols / 4, THREAD_COUNT);

#pragma omp parallel for collapse(2) num_threads(THREAD_COUNT)
	for (int i = 0; i < rows / 4; i++) {
		for (int j = 0; j < cols / 4; j++) {
			map[i][j] += feat36[i][j] + feat37[i][j];
		}
	}

	nrm(feat47, rows / 4, cols / 4, THREAD_COUNT);
	nrm(feat48, rows / 4, cols / 4, THREAD_COUNT);

#pragma omp parallel for collapse(2) num_threads(THREAD_COUNT)
	for (int i = 0; i < rows / 4; i++) {
		for (int j = 0; j < cols / 4; j++) {
			map[i][j] += feat47[i][j] + feat48[i][j];
		}
	}
}


void SaliencyMapGPU::centerSurroundDiff(double** &supLevel, double** &lowLevel, double ** &difference, int sup, int low, int endl) {
	int supRow = rows / pow2(sup);
	int supCol = cols / pow2(sup);

	int lowRow = rows / pow2(low);
	int lowCol = cols / pow2(low);

	double **growLowLevel = allocate(supRow, supCol);
	FilterGPU::growthMatrix(lowLevel, lowRow, lowCol, growLowLevel, pow2(low - sup), THREAD_COUNT);

	if (sup != endl) {
		double **rawDifference = allocate(supRow, supCol);

		absDifference(rawDifference, supLevel, growLowLevel, supRow, supCol);
		FilterGPU::growthMatrix(rawDifference, supRow, supCol, difference, pow2(sup - endl), THREAD_COUNT);

		FilterGPU::deleteMemory(rawDifference, supRow, supCol);
	}
	else {
		absDifference(difference, supLevel, growLowLevel, supRow, supCol);
	}

	FilterGPU::deleteMemory(growLowLevel, supRow, supCol);
}

void SaliencyMapGPU::absDifference(double** out, double** first, double** second, int rows, int cols) {
	double a, b;

#pragma omp parallel for collapse(2) num_threads(THREAD_COUNT)
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			a = first[i][j];
			b = second[i][j];
			out[i][j] = (a > b) ? (a - b) : (b - a);
		}
	}
}


void SaliencyMapGPU::getSalency() {
	int i,j,ip,jp;

	int local_pad1R = pad1R/4;
	int local_pad1C = pad1C/4;
	int local_pad2R = pad2R/4;
	int local_pad2C = pad2C/4;

	int local_rows = local_pad2R-local_pad1R;
	int local_cols = local_pad2C-local_pad1C;

	_Salency = allocate(local_rows, local_cols);

	for (i=local_pad1R, ip=0; i < local_pad2R; ++i, ++ip) {
		for (j=local_pad1C, jp=0; j < local_pad2C; ++j, ++jp) {
			_Salency[ip][jp] = _Imap[i][j] + _Omap[i][j] + _Cmap[i][j];
		}
	}
}

void SaliencyMapGPU::showSalency(){
	int local_rows = (pad2R/4)-(pad1R/4);
	int local_cols = (pad2C/4)-(pad1C/4);

	double maxImg, minImg, coeff;

	maxArray(_Salency, maxImg, local_rows, local_cols, THREAD_COUNT);
	minArray(_Salency, minImg, local_rows, local_cols, THREAD_COUNT);

	coeff = 255 / (maxImg - minImg);

	Mat out = cv::Mat(local_rows, local_cols, CV_8UC1);

	for (int i = 0; i < local_rows; i++) {
		for (int j = 0; j < local_cols; j++) {
			out.at<uchar>(i, j) = (uchar)(coeff*(_Salency[i][j] - minImg));
		}
	}

	resize(out, out, Size(), 2, 2, INTER_LANCZOS4);

	SaliencyMapGPU::imshow(out, "Salency Map!!");
}



void SaliencyMapGPU::imshow(double **img, int x_length, int y_length, std::string name = "Una ventana") {
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

	SaliencyMapGPU::imshow(out, name);
}


void SaliencyMapGPU::imshow(Mat img, std::string name = "Una ventana") {
	cv::imshow(name, img);
	waitKey(0);
}
