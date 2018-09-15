//包含巴特沃斯滤波的函数与变量
#pragma once
#include "stdafx.h"
#include<stdlib.h>
#include <opencv2/opencv.hpp> 
#include <iostream> 
#include<math.h>

//定义自定义的dft函数功能
#define CUSTOM_DFT_SHOW 0//输出为浮点型可显示的图像，并作对数变换以增强图像对比度
#define CUSTOM_DFT_RAW 1//输出为浮点型可显示图像，不作对数变换
#define CUSTOM_DFT_COMPLEX 2//输出为dft的复数结果，用于进一步的频率域变换

#define CUSTOM_BUTTERWORTH_COMPLEX 0//输出为复数类型矩阵
#define CUSTOM_BUTTERWORTH_SHOW 1//输出为浮点型图像

//定义频谱中的八个冲激点的图像坐标，此处取原点为图像中心
/*
#define UK1 -39
#define VK1 -32
#define UK2 -80
#define VK2 -32
#define UK3 -43
#define VK3 29
#define UK4 -85
#define VK4 29
*/

using namespace cv;
using namespace std;

void dft_image(Mat& src, Mat& dst, int flag);//自定义的dft函数，输入为原始图像，输出为浮点型图像或复数类型矩阵
void idft_image(Mat& srcComplex, Mat& dstReal);//自定义的idft函数，输入为复数类型矩阵，输出为浮点型实矩阵
void idft_image(Mat& srcComplex, Mat& dstReal, int width, int height);//重载idft_image函数，对结果图像的0填充部分进行切除，输入为原图像的宽与高
void perform_butterworth(Mat& srcComplex, Mat& dstComplex, vector<int> coordinates,  double D0, int n, int flag);//对图像进行巴特沃斯滤波的主函数
double Dk(int u, int v, int uk, int vk, int M, int N);//滤波器函数中Dk项计算
double Dk_minus(int u, int v, int uk, int vk, int M, int N);//滤波器函数中D-k项计算
double Butterworth_per_pixel(double D0, int n, double dk, double dk_minus);//图像中对每个像素点的巴特沃斯表达式计算
void get_filter(Mat& srcComplex, Mat& dst, vector<int> coordinaates, double D0, int n);//返回频率域的滤波器图像，为浮点类型
vector<int> get_impulse_points(Mat& src, int radius);

void dft_image(Mat& src, Mat& dst, int flag = 0) {
	Mat padded;//进行0填充的图像矩阵
	//得到进行傅里叶变换后的图像大小
	int m = getOptimalDFTSize(src.rows);
	int n = getOptimalDFTSize(src.cols);
	//进行0填充
	copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
	//初始化含有两个Mat变量的数组，分别存储结果复数矩阵的实部和虚部
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32FC1) };
	Mat complexI;
	//将complexI扩展为二维矩阵
	merge(planes, 2, complexI);
	//执行dft运算
	dft(complexI, complexI);
	//将complexI的两个维度（实部和虚部）分别存储在planes[0]和planes[1]中
	split(complexI, planes);
	//获得幅值并存储在planes[0]中
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];
	if (flag == 1) {//输出平移并归一化后的图像
		//将图像的一三象限以及二四象限对换，使得频谱中心位于图像中心
		int cx = magI.cols / 2;
		int cy = magI.rows / 2;
		//设置ROI
		Mat q0 = Mat(magI, Rect(0, 0, cx, cy));
		Mat q1 = Mat(magI, Rect(cx, 0, cx, cy));
		Mat q2 = Mat(magI, Rect(0, cy, cx, cy));
		Mat q3 = Mat(magI, Rect(cx, cy, cx, cy));

		Mat tmp;
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);
		q2.copyTo(q1);
		tmp.copyTo(q2);
		//图像归一化
		cv::normalize(magI, magI, 0, 1, CV_MINMAX);
		dst = magI;
	}
	else if (flag == 2) {//输出复数矩阵，仅作图像象限平移
		int cx = complexI.cols / 2;
		int cy = complexI.rows / 2;
		Mat q0 = Mat(complexI, Rect(0, 0, cx, cy));
		Mat q1 = Mat(complexI, Rect(cx, 0, cx, cy));
		Mat q2 = Mat(complexI, Rect(0, cy, cx, cy));
		Mat q3 = Mat(complexI, Rect(cx, cy, cx, cy));

		Mat tmp;
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);
		q2.copyTo(q1);
		tmp.copyTo(q2);
		dst = complexI;
	}
	else {//输出增强后的频谱图像，添加了对数变换步骤
		magI += Scalar::all(1);
		log(magI, magI);
		int cx = magI.cols / 2;
		int cy = magI.rows / 2;
		Mat q0 = Mat(magI, Rect(0, 0, cx, cy));
		Mat q1 = Mat(magI, Rect(cx, 0, cx, cy));
		Mat q2 = Mat(magI, Rect(0, cy, cx, cy));
		Mat q3 = Mat(magI, Rect(cx, cy, cx, cy));

		Mat tmp;
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);

		q1.copyTo(tmp);
		q2.copyTo(q1);
		tmp.copyTo(q2);
		cv::normalize(magI, magI, 0, 1, CV_MINMAX);
		dst = magI;
	}
}

void idft_image(Mat& srcComplex, Mat& dstReal) {
	//先对图像象限进行对调
	int cx = srcComplex.cols / 2;
	int cy = srcComplex.rows / 2;
	Mat q0 = Mat(srcComplex, Rect(0, 0, cx, cy));
	Mat q1 = Mat(srcComplex, Rect(cx, 0, cx, cy));
	Mat q2 = Mat(srcComplex, Rect(0, cy, cx, cy));
	Mat q3 = Mat(srcComplex, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//得到归一化后的结果图像
	Mat planes[] = { Mat::zeros(srcComplex.size(), CV_32F), Mat::zeros(srcComplex.size(), CV_32F) };
	idft(srcComplex, srcComplex);
	split(srcComplex, planes);
	magnitude(planes[0], planes[1], planes[0]);
	cv::normalize(planes[0], planes[0], 0, 1, CV_MINMAX);
	dstReal = planes[0];
}

void idft_image(Mat& srcComplex, Mat& dstReal, int width, int height) {
	//先对图像象限进行对调
	int cx = srcComplex.cols / 2;
	int cy = srcComplex.rows / 2;
	Mat q0 = Mat(srcComplex, Rect(0, 0, cx, cy));
	Mat q1 = Mat(srcComplex, Rect(cx, 0, cx, cy));
	Mat q2 = Mat(srcComplex, Rect(0, cy, cx, cy));
	Mat q3 = Mat(srcComplex, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//得到归一化后的结果图像
	Mat planes[] = { Mat::zeros(srcComplex.size(), CV_32F), Mat::zeros(srcComplex.size(), CV_32F) };
	idft(srcComplex, srcComplex);
	split(srcComplex, planes);
	magnitude(planes[0], planes[1], planes[0]);
	cv::normalize(planes[0], planes[0], 0, 1, CV_MINMAX);
	dstReal = Mat(planes[0], Rect(0, 0, width, height));
}

void get_filter(Mat& srcComplex, Mat& dst, vector<int> coordinates, double D0, int n) {
	//定义频域滤波器图像
	Mat filter = Mat(srcComplex.size(), CV_32F);
	for (int i = 0; i < srcComplex.rows; i++) {
		for (int j = 0; j < srcComplex.cols; j++) {
			//遍历图像，对每个像素点计算Butterworth滤波器对应的值，在原图像冲激附近滤波器值接近0，而在其它位置滤波器值接近1
			float result = 1.0;

			for (int k = 0; k < coordinates.size(); k += 2) {
				double dk1 = Dk(i, j, coordinates[k], coordinates[k + 1], srcComplex.rows, srcComplex.cols);
				double dk1_m = Dk_minus(i, j, coordinates[k], coordinates[k + 1], srcComplex.rows, srcComplex.cols);
				double h1 = Butterworth_per_pixel(D0, n, dk1, dk1_m);
				result *= h1;
			}
			/*
			double dk1 = Dk(i, j, UK1, VK1, srcComplex.rows, srcComplex.cols);
			double dk1_m = Dk_minus(i, j, UK1, VK1, srcComplex.rows, srcComplex.cols);
			double h1 = Butterworth_per_pixel(D0, n, dk1, dk1_m);
			result *= h1;

			double dk2 = Dk(i, j, UK2, VK2, srcComplex.rows, srcComplex.cols);
			double dk2_m = Dk_minus(i, j, UK2, VK2, srcComplex.rows, srcComplex.cols);
			double h2 = Butterworth_per_pixel(D0, n, dk2, dk2_m);
			result *= h2;

			double dk3 = Dk(i, j, UK3, VK3, srcComplex.rows, srcComplex.cols);
			double dk3_m = Dk_minus(i, j, UK3, VK3, srcComplex.rows, srcComplex.cols);
			double h3 = Butterworth_per_pixel(D0, n, dk3, dk3_m);
			result *= h3;

			double dk4 = Dk(i, j, UK4, VK4, srcComplex.rows, srcComplex.cols);
			double dk4_m = Dk_minus(i, j, UK4, VK4, srcComplex.rows, srcComplex.cols);
			double h4 = Butterworth_per_pixel(D0, n, dk4, dk4_m);
			result *= h4;
			*/
			filter.at<float>(i, j) = result;
		}
	}
	dst = filter;
}

void perform_butterworth(Mat& srcComplex, Mat& dstComplex, vector<int> coordinates,double D0, int n, int flag = 0) {
	Mat filter = Mat(srcComplex.size(), CV_32F);
	for (int i = 0; i < srcComplex.rows; i++) {
		for (int j = 0; j < srcComplex.cols; j++) {
			float result = 1.0;

			for (int k = 0; k < coordinates.size(); k += 2) {
				double dk1 = Dk(i, j, coordinates[k], coordinates[k+1], srcComplex.rows, srcComplex.cols);
				double dk1_m = Dk_minus(i, j, coordinates[k], coordinates[k+1], srcComplex.rows, srcComplex.cols);
				double h1 = Butterworth_per_pixel(D0, n, dk1, dk1_m);
				result *= h1;
			}
			filter.at<float>(i, j) = result;
		}
	}
	Mat planes[] = { Mat::zeros(srcComplex.size(), CV_32F), Mat::zeros(srcComplex.size(), CV_32F) };
	split(srcComplex, planes);
	//将复数矩阵的实部和虚部分别和滤波器图像相乘得到结果的频率域图像
	planes[0] = planes[0].mul(filter);
	planes[1] = planes[1].mul(filter);
	merge(planes, 2, dstComplex);

	if (flag == 1) {
		//对结果图像进行提取幅值，对数变换与归一化操作
		magnitude(planes[0], planes[1], planes[0]);
		Mat magI = planes[0];
		magI += Scalar::all(1);
		log(magI, magI);
		cv::normalize(magI, magI, 0, 1, CV_MINMAX);
		dstComplex = magI;
	}
}

double Dk(int u, int v, int uk, int vk, int M, int N) {
	return sqrt(pow((double)(u - M / 2 - uk), 2.0) + pow((double)(v - N / 2 - vk), 2.0));
}

double Dk_minus(int u, int v, int uk, int vk, int M, int N) {
	return sqrt(pow((double)(u - M / 2 + uk), 2.0) + pow((double)(v - N / 2 + vk), 2.0));
}

double Butterworth_per_pixel(double D0, int n, double dk, double dk_minus) {
	return (1.0 / (1.0 + pow(((double)D0 / (double)dk), 2.0 * (double)n))) * (1.0 / (1.0 + pow(((double)D0 / (double)dk_minus), 2.0 * (double)n)));
}

vector<int> get_impulse_points(Mat& src, int radius = 18){
	GaussianBlur(src, src, Size(3, 3), 0, 0);
	threshold(src, src, 0.5, 1.0, CV_THRESH_BINARY);

	Mat q1 = Mat(src, Rect(0, 0, src.cols / 2 - radius, src.rows / 2 - radius));
	cv::normalize(q1, q1, 0, 255, CV_MINMAX);
	q1.convertTo(q1, CV_8U);

	vector<Point> central_points;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<int>coordinates;

	cv::findContours(q1, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	int x_min = 500;
	int x_max = 0;
	int y_min = 500;
	int y_max = 0;
	Point p_tmp;
	for (int i = 0; i<contours.size(); i++)
	{
		//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数  
		for (int j = 0; j<contours[i].size(); j++)
		{
			//找到contours向量内所有的像素点  
			Point P = Point(contours[i][j].x, contours[i][j].y);
			if (P.x < x_min) {
				x_min = P.x;
			}
			if (P.x > x_max) {
				x_max = P.x;
			}
			if (P.y < y_min) {
				y_min = P.y;
			}
			if (P.y > y_max) {
				y_max = P.y;
			}
		}
		if ((x_max - x_min) * (y_max - y_min) >= 4) {
			p_tmp = Point((x_min + x_max) / 2, (y_min + y_max) / 2);
			central_points.push_back(p_tmp);
			coordinates.push_back(p_tmp.y - src.rows / 2);
			coordinates.push_back(p_tmp.x - src.cols / 2);
		}
		else {
			if (src.rows / 2 - y_max > 40) {
				p_tmp = Point((x_min + x_max) / 2, (y_min + y_max) / 2);
				central_points.push_back(p_tmp);
				coordinates.push_back(p_tmp.y - src.rows / 2);
				coordinates.push_back(p_tmp.x - src.cols / 2);
			}
		}
		x_min = 500;
		x_max = 0;
		y_min = 500;
		y_max = 0;
	}

	Mat q2 = Mat(src, Rect(src.cols / 2 + radius, 0, src.cols / 2 - radius, src.rows / 2 - radius));
	cv::normalize(q2, q2, 0, 255, CV_MINMAX);
	q2.convertTo(q2, CV_8U);

	cv::findContours(q2, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	for (int i = 0; i<contours.size(); i++)
	{
		//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数  
		for (int j = 0; j<contours[i].size(); j++)
		{
			//找到contours向量内所有的像素点  
			Point P = Point(contours[i][j].x, contours[i][j].y);
			if (P.x < x_min) {
				x_min = P.x;
			}
			if (P.x > x_max) {
				x_max = P.x;
			}
			if (P.y < y_min) {
				y_min = P.y;
			}
			if (P.y > y_max) {
				y_max = P.y;
			}
		}
		if ((x_max - x_min) * (y_max - y_min) >= 4) {
			p_tmp = Point((x_min + x_max) / 2, (y_min + y_max) / 2);
			central_points.push_back(p_tmp);
			coordinates.push_back(p_tmp.y - src.rows / 2);
			coordinates.push_back(p_tmp.x + radius);
		}
		else {
			if (src.rows / 2 - y_max > 40) {
				p_tmp = Point((x_min + x_max) / 2, (y_min + y_max) / 2);
				central_points.push_back(p_tmp);
				coordinates.push_back(p_tmp.y - src.rows / 2);
				coordinates.push_back(p_tmp.x + radius);
			}
		}
		x_min = 500;
		x_max = 0;
		y_min = 500;
		y_max = 0;
	}

	
	return coordinates;
}