#include "stdafx.h"
#include "ButterWorth.h"

using namespace cv;      
using namespace std;   

int main(int argc, char** argv) {   
	Mat src = imread("D:\\Custom\\opencv\\ButterWorth\\test.tif");   
	if (src.empty()) {                  
		printf("could not load image...\n");
		waitKey(0);
		system("PAUSE");
		return -1;
	}
	cvtColor(src, src, CV_BGR2GRAY);

	Mat srcCopy;
	src.copyTo(srcCopy);

	Mat frequency;
	dft_image(src, frequency, CUSTOM_DFT_SHOW);
	
	Mat frequency_copy;
	frequency.copyTo(frequency_copy);
	vector<int> central_points;
	central_points = get_impulse_points(frequency_copy, 18);

	for (int i = 0; i < central_points.size(); i+=2) {
		cout << central_points[i] << " " << central_points[i+1] << endl;
	}

	dft_image(src, src, CUSTOM_DFT_COMPLEX);
	
	Mat magI;
	perform_butterworth(src, magI, central_points, 10.0, 6, CUSTOM_BUTTERWORTH_SHOW);

	perform_butterworth(src, src, central_points, 10.0, 6);

	Mat filter = Mat(src.size(), CV_32F);
	get_filter(src, filter, central_points, 10.0, 6);
	cv::normalize(filter, filter, 0, 255, CV_MINMAX);
	filter.convertTo(filter, CV_8U);

	Mat result;
	idft_image(src, result, srcCopy.cols, srcCopy.rows);

	imshow("Car", srcCopy);
	imshow("Original", frequency);
	imshow("Processed", frequency_copy);
	imshow("Result", magI); 
	imshow("Filter", filter);
	imshow("Image", result);
	waitKey(0);
	system("PAUSE");
	return 0;
}

