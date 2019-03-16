#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
const char* keys =
        "{ help h |                          | Print help message. }"
        "{ input1 | ../data/box.png          | Path to input image 1. }"
        "{ input2 | ../data/box_in_scene.png | Path to input image 2. }";

int main()
{
    // VideoCapture capture("../data/tree.avi");
    VideoCapture capture(0);
    if (!capture.isOpened())
    {
        std::cout << "读取失败" << std::endl;
        return 1;
    }
    double rate = capture.get(CAP_PROP_FPS);
    cout << "帧率：" << rate << endl;
    bool stop(false);
    Mat frame;
    int delay = 1000 / rate;
    namedWindow("Extracted Frame");
    while (!stop){
//        if (!capture.read(frame))
//            break;
        capture >> frame;
            //break;
        imshow("Extracted Frame", frame);
        if(waitKey(delay) > 0)
            stop = true;
    }
    waitKey();
    return 0;
}



