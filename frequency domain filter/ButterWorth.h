//����������˹�˲��ĺ��������
#pragma once
#include "stdafx.h"
#include<stdlib.h>
#include <opencv2/opencv.hpp> 
#include <iostream> 
#include<math.h>

//�����Զ����dft��������
#define CUSTOM_DFT_SHOW 0//���Ϊ�����Ϳ���ʾ��ͼ�񣬲��������任����ǿͼ��Աȶ�
#define CUSTOM_DFT_RAW 1//���Ϊ�����Ϳ���ʾͼ�񣬲��������任
#define CUSTOM_DFT_COMPLEX 2//���Ϊdft�ĸ�����������ڽ�һ����Ƶ����任

#define CUSTOM_BUTTERWORTH_COMPLEX 0//���Ϊ�������;���
#define CUSTOM_BUTTERWORTH_SHOW 1//���Ϊ������ͼ��

//����Ƶ���еİ˸��弤���ͼ�����꣬�˴�ȡԭ��Ϊͼ������
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

void dft_image(Mat& src, Mat& dst, int flag);//�Զ����dft����������Ϊԭʼͼ�����Ϊ������ͼ��������;���
void idft_image(Mat& srcComplex, Mat& dstReal);//�Զ����idft����������Ϊ�������;������Ϊ������ʵ����
void idft_image(Mat& srcComplex, Mat& dstReal, int width, int height);//����idft_image�������Խ��ͼ���0��䲿�ֽ����г�������Ϊԭͼ��Ŀ����
void perform_butterworth(Mat& srcComplex, Mat& dstComplex, vector<int> coordinates,  double D0, int n, int flag);//��ͼ����а�����˹�˲���������
double Dk(int u, int v, int uk, int vk, int M, int N);//�˲���������Dk�����
double Dk_minus(int u, int v, int uk, int vk, int M, int N);//�˲���������D-k�����
double Butterworth_per_pixel(double D0, int n, double dk, double dk_minus);//ͼ���ж�ÿ�����ص�İ�����˹���ʽ����
void get_filter(Mat& srcComplex, Mat& dst, vector<int> coordinaates, double D0, int n);//����Ƶ������˲���ͼ��Ϊ��������
vector<int> get_impulse_points(Mat& src, int radius);

void dft_image(Mat& src, Mat& dst, int flag = 0) {
	Mat padded;//����0����ͼ�����
	//�õ����и���Ҷ�任���ͼ���С
	int m = getOptimalDFTSize(src.rows);
	int n = getOptimalDFTSize(src.cols);
	//����0���
	copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
	//��ʼ����������Mat���������飬�ֱ�洢������������ʵ�����鲿
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32FC1) };
	Mat complexI;
	//��complexI��չΪ��ά����
	merge(planes, 2, complexI);
	//ִ��dft����
	dft(complexI, complexI);
	//��complexI������ά�ȣ�ʵ�����鲿���ֱ�洢��planes[0]��planes[1]��
	split(complexI, planes);
	//��÷�ֵ���洢��planes[0]��
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];
	if (flag == 1) {//���ƽ�Ʋ���һ�����ͼ��
		//��ͼ���һ�������Լ��������޶Ի���ʹ��Ƶ������λ��ͼ������
		int cx = magI.cols / 2;
		int cy = magI.rows / 2;
		//����ROI
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
		//ͼ���һ��
		cv::normalize(magI, magI, 0, 1, CV_MINMAX);
		dst = magI;
	}
	else if (flag == 2) {//����������󣬽���ͼ������ƽ��
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
	else {//�����ǿ���Ƶ��ͼ������˶����任����
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
	//�ȶ�ͼ�����޽��жԵ�
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
	//�õ���һ����Ľ��ͼ��
	Mat planes[] = { Mat::zeros(srcComplex.size(), CV_32F), Mat::zeros(srcComplex.size(), CV_32F) };
	idft(srcComplex, srcComplex);
	split(srcComplex, planes);
	magnitude(planes[0], planes[1], planes[0]);
	cv::normalize(planes[0], planes[0], 0, 1, CV_MINMAX);
	dstReal = planes[0];
}

void idft_image(Mat& srcComplex, Mat& dstReal, int width, int height) {
	//�ȶ�ͼ�����޽��жԵ�
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
	//�õ���һ����Ľ��ͼ��
	Mat planes[] = { Mat::zeros(srcComplex.size(), CV_32F), Mat::zeros(srcComplex.size(), CV_32F) };
	idft(srcComplex, srcComplex);
	split(srcComplex, planes);
	magnitude(planes[0], planes[1], planes[0]);
	cv::normalize(planes[0], planes[0], 0, 1, CV_MINMAX);
	dstReal = Mat(planes[0], Rect(0, 0, width, height));
}

void get_filter(Mat& srcComplex, Mat& dst, vector<int> coordinates, double D0, int n) {
	//����Ƶ���˲���ͼ��
	Mat filter = Mat(srcComplex.size(), CV_32F);
	for (int i = 0; i < srcComplex.rows; i++) {
		for (int j = 0; j < srcComplex.cols; j++) {
			//����ͼ�񣬶�ÿ�����ص����Butterworth�˲�����Ӧ��ֵ����ԭͼ��弤�����˲���ֵ�ӽ�0����������λ���˲���ֵ�ӽ�1
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
	//�����������ʵ�����鲿�ֱ���˲���ͼ����˵õ������Ƶ����ͼ��
	planes[0] = planes[0].mul(filter);
	planes[1] = planes[1].mul(filter);
	merge(planes, 2, dstComplex);

	if (flag == 1) {
		//�Խ��ͼ�������ȡ��ֵ�������任���һ������
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
		//contours[i]������ǵ�i��������contours[i].size()������ǵ�i�����������е����ص���  
		for (int j = 0; j<contours[i].size(); j++)
		{
			//�ҵ�contours���������е����ص�  
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
		//contours[i]������ǵ�i��������contours[i].size()������ǵ�i�����������е����ص���  
		for (int j = 0; j<contours[i].size(); j++)
		{
			//�ҵ�contours���������е����ص�  
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