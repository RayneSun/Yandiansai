#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "stereoRectified.h"

using namespace cv;

StereoRetifier::StereoRetifier(double *_camMatrixLeft, double *_camMatrixRight, double *_distCoeffLeft,
                               double *_distCoeffRight, double *_R, double *_T) {
    camMatrixLeft.create(3, 3, CV_64FC1);
    camMatrixRight.create(3, 3, CV_64FC1);

    distCoeffLeft.create(4, 1, CV_64FC1);
    distCoeffRight.create(4, 1, CV_64FC1);

    R.create(3, 3, CV_64FC1);
    T.create(3, 1, CV_64FC1);

    RectRotLeft.create(3, 3, CV_64FC1);
    RectRotRight.create(3, 3, CV_64FC1);

    newCamMatrixLeft.create(3, 4, CV_64FC1);
    newCamMatrixRight.create(3, 4, CV_64FC1);

    Q.create(4, 4, CV_64FC1);

    memcpy(camMatrixLeft.data, _camMatrixLeft, sizeof(double) * 3 * 3);
    memcpy(camMatrixRight.data, _camMatrixRight, sizeof(double) * 3 * 3);

    memcpy(distCoeffLeft.data, _distCoeffLeft, sizeof(double) * 4);
    memcpy(distCoeffRight.data, _distCoeffRight, sizeof(double) * 4);

    memcpy(R.data, _R, 3 * 3 * sizeof(double));
    memcpy(T.data, _T, 3 * sizeof(double));
}

void StereoRetifier::process(Mat leftRaw, Mat rightRaw, Mat &left, Mat &right) {
    remap(leftRaw, left, map1Left, map2Left, INTER_CUBIC);
    remap(rightRaw, right, map1Right, map2Right, INTER_CUBIC);

}

void StereoRetifier::init(int width, int height) {
    stereoRectify(camMatrixLeft, distCoeffLeft, camMatrixRight, distCoeffRight,
                  Size(width, height), R, T, RectRotLeft, RectRotRight,
                  newCamMatrixLeft, newCamMatrixRight, Q, CALIB_ZERO_DISPARITY,
                  0, Size(0, 0), &ROILeft, &ROIRight);

    initUndistortRectifyMap(camMatrixLeft, distCoeffLeft, RectRotLeft, newCamMatrixLeft, Size(width, height), CV_32FC1,
                            map1Left, map2Left);

    initUndistortRectifyMap(camMatrixRight, distCoeffRight, RectRotRight, newCamMatrixRight, Size(width, height),
                            CV_32FC1, map1Right, map2Right);

}

StereoRetifier::~StereoRetifier() {

}


void Rectifier(StereoRetifier stir, Mat &left_frame, Mat &right_frame,Mat &leftRectified,Mat &rightRectified) {

    stir.init(left_frame.cols, right_frame.rows);
    stir.process(left_frame, right_frame, leftRectified, rightRectified);

    //划线
//    Mat leftRightMerge;
//    leftRightMerge.create(leftRectified.rows, leftRectified.cols + rightRectified.cols, CV_8UC1);
//    leftRectified.copyTo(leftRightMerge(Rect(0, 0, leftRectified.cols, leftRectified.rows)));
//    rightRectified.copyTo(leftRightMerge(Rect(leftRectified.cols, 0, rightRectified.cols, rightRectified.rows)));
//    //画线
//    for (int a = 0; a < leftRightMerge.rows; a += 30) {
//        line(leftRightMerge, Point(0, i), Point(leftRightMerge.cols - 1, i), Scalar(128), 3);
//    }

}
