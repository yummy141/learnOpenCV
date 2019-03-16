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
const char* myWindow = "This is a sift test!";


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
    Mat preFrame, nowFrame;
    int delay = 1000 / rate;

    // Window
    namedWindow(myWindow, WINDOW_NORMAL);

    // capture the first frame
    capture >> preFrame;
    waitKey(delay);

    while (!stop){
        if (!capture.read(nowFrame))
            break;
        // capture >> frame;
            // break;

        // sift
        //-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
        Ptr<SIFT> detector = SIFT::create();
        std::vector<KeyPoint> keypoints_preFrame, keypoints_nowFrame;
        Mat descriptors_preFrame, descriptors_nowFrame;
        detector->detectAndCompute( preFrame, noArray(), keypoints_preFrame, descriptors_preFrame );
        detector->detectAndCompute( nowFrame, noArray(), keypoints_nowFrame, descriptors_nowFrame );

        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors_preFrame, descriptors_nowFrame, knn_matches, 2 );
        //-- Filter matches using the Lowe's ratio test


        const float ratio_thresh = 0.75f;
        // double dMyRatihoThresh =  static_cast<double>(myRatioThresh)/100.0;


        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
                good_matches.push_back(knn_matches[i][0]);

        //-- Draw matches
        Mat img_matches;
        drawMatches( preFrame, keypoints_preFrame, nowFrame, keypoints_nowFrame, good_matches, img_matches, Scalar::all(-1),
                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        resizeWindow(myWindow, 1600, 500);
        imshow(myWindow, img_matches);
        preFrame = nowFrame;
        if(waitKey(delay) > 0)
            stop = true;
    }
    waitKey();
    return 0;
}



