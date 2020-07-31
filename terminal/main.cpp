#include <opencv2/opencv.hpp>
#include <iostream>
#include "stereoRectified.h"
#include "circle.h"

//      相机内参
double camMatrixLeft[9] = {613.3814404, -2.705110705, 316.5671179,
                           0, 614.4412715, 236.3111183,
                           0, 0, 1,
};

double camMatrixRight[9] = {607.1091988, -0.894738312, 316.6710273,
                            0, 609.966985, 247.422388,
                            0, 0, 1
};

//      相机畸变参数
double distCoeffLeft[4] = {-0.443980476, 0.252314209, 0.002885095, 0.00070709
};   // small_2的相机参数
double distCoeffRight[4] = {-0.42788248, 0.144356954, 0.001962326, -0.002715673
};  // small_2的相机参数
double T[3] = {-98.77344858, -0.082151978, -5.119274898,
};

double R[9] = {0.999759792, -0.019056445, 0.010826339,
               0.019083495, 0.999815011, -0.002400784,
               -0.010778586, 0.002606812, 0.999938511


};
const unsigned int fps = 100;

int main() {


    // 读取Usb相机视频
    VideoCapture capture1(2);
    VideoCapture capture2(1);
    if ((!capture1.isOpened()) || (!capture2.isOpened())) {
        cerr << "open error \n" << endl;
        return 1;
    }

//////    测试代码段OK，配合里面代码段magic
//    int i = 1;
//    char str[256];
//    char str2[256];


    Mat left_frame, right_frame;
    double distance;
    vector<Vec3f> left_circle_batch, right_circle_batch;
    Vec3f left_circle, right_circle;
    StereoRetifier stereoRectifier(camMatrixLeft, camMatrixRight, distCoeffLeft, distCoeffRight, R, T);

    while (true) {

//此处添加读取摄像头函数封装。。
        capture2 >> right_frame;
        capture1 >> left_frame;
        cvtColor(right_frame, right_frame, CV_BGR2GRAY);
        cvtColor(left_frame, left_frame, CV_BGR2GRAY);


////测试代码段magic，配合上面代码段ok
//        sprintf(str, "D:/Sample/left/left_%d.jpg", i);
//        sprintf(str2, "D:/Sample/right/right_%d.jpg", i);
//        left_frame = imread(str, 0);
//        right_frame = imread(str2, 0);
//        i++;


        if (left_frame.empty()) {
            cout << "left_frame is empty" << endl;
            return 1;
        } else if (right_frame.empty()) {
            cout << "right_frame is empty" << endl;
            return 1;
        }

        Mat leftRectified(left_frame.rows, left_frame.cols, CV_8UC1, Scalar(255));
        Mat rightRectified(right_frame.rows, right_frame.cols, CV_8UC1, Scalar(255));


        Rectifier(stereoRectifier, left_frame, right_frame, leftRectified, rightRectified);


        detectCircle(leftRectified, left_circle_batch);
        detectCircle(rightRectified, right_circle_batch);

        //切换为BGR便于显示
        cvtColor(leftRectified, leftRectified, COLOR_GRAY2BGR);
//        cvtColor(rightRectified, rightRectified, COLOR_GRAY2BGR);

        bool isEmpty = circleFilter(left_circle_batch, right_circle_batch, left_circle, right_circle);
        if (isEmpty) {
            putText(leftRectified, "Cannot find target", textpoint, 2, 1.8, color, 3);
//            putText(rightRectified, "Cannot find circle", textpoint, 2, 0.5, color, 2);
            imshow("Leftcircle", leftRectified);
//            imshow("Rightcircle", rightRectified);
            waitKey(fps);
        } else {
            distance = (618 * 98.604) / (abs(left_circle[0] - right_circle[0]));
            drawCircle(leftRectified, left_circle, "Leftcircle", distance / 10.0);
//            drawCircle(rightRectified, right_circle, "Rightcircle", distance);
            waitKey(fps);
        }
    }
}






























