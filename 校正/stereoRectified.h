#ifndef __STE_H
#define __STE_H
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;



class StereoRetifier
{
public:
	Mat camMatrixLeft;
	Mat camMatrixRight;

	Mat distCoeffLeft;
	Mat distCoeffRight;

	Mat map1Left;
	Mat map1Right;
	Mat map2Left;
	Mat map2Right;

	Mat R;
	Mat T;
	Mat Q;
	Mat RectRotLeft;
	Mat RectRotRight;

	Mat newCamMatrixLeft;
	Mat newCamMatrixRight;

	Rect ROILeft;
	Rect ROIRight;

	StereoRetifier(double* _camMatrixLeft,double* _camMatrixRight, double *_distCoeffLeft, double *_distCoeffRight, double *_R,double *_T);
	~StereoRetifier();
	void init(int width ,int height);
	void process(Mat leftRaw, Mat rightRaw, Mat &left ,Mat &right);

};


void Rectifier(StereoRetifier stir, Mat &left_frame, Mat &right_frame,Mat &leftRectified,Mat &rightRectified);


#endif