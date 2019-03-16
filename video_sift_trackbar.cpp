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

const char* myWindow = "This is a sift test!";
void mySift_function( int, void* );
// for sift
int myNfeatures = 0, maxNfeatures = 200;
int myOctaveLayers = 3, maxOctaveLayers = 20;
int myContrastThreshold = 10, maxContrastThreshold = 30;
int myEdgeThreshold = 20, maxEdgeThreshold = 100;
int mySigma = 16, maxSigma = 50;

double dMyContrastThreshold, dMyEdgeThreshold, dMySigma;
double dMyRatihoThresh;
// for Ratio_thresh
int myRatioThresh = 75, maxRatioThresh = 100;

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
    createTrackbar( "nfeatures:", myWindow, &myNfeatures, maxNfeatures, mySift_function );
    createTrackbar( "nOctaveLayers:", myWindow, &myOctaveLayers, maxOctaveLayers, mySift_function );
    createTrackbar( "contrastThreshold(*100):", myWindow, &myContrastThreshold, maxContrastThreshold, mySift_function );
    createTrackbar( "edgeThreshold(*100) :", myWindow, &myEdgeThreshold, maxEdgeThreshold, mySift_function );
    createTrackbar( "sigma(*10):", myWindow, &mySigma, maxSigma, mySift_function );
    createTrackbar( "ratio_thresh(*100):", myWindow, &myRatioThresh, maxRatioThresh, mySift_function );

    mySift_function(0, 0);
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
        Ptr<SIFT> detector = SIFT::create(myNfeatures, myOctaveLayers, dMyContrastThreshold, dMyEdgeThreshold, dMySigma);
        std::vector<KeyPoint> keypoints_preFrame, keypoints_nowFrame;
        Mat descriptors_preFrame, descriptors_nowFrame;
        detector->detectAndCompute( preFrame, noArray(), keypoints_preFrame, descriptors_preFrame );
        detector->detectAndCompute( nowFrame, noArray(), keypoints_nowFrame, descriptors_nowFrame );

        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors_preFrame, descriptors_nowFrame, knn_matches, 2 );
        //-- Filter matches using the Lowe's ratio test


        // const float ratio_thresh = 0.75f;
        // double dMyRatihoThresh =  static_cast<double>(myRatioThresh)/100.0;


        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
            if (knn_matches[i][0].distance < dMyRatihoThresh * knn_matches[i][1].distance)
                good_matches.push_back(knn_matches[i][0]);

        //-- Draw matches
        Mat img_matches;
        drawMatches( preFrame, keypoints_preFrame, nowFrame, keypoints_nowFrame, good_matches, img_matches, Scalar::all(-1),
                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        resizeWindow(myWindow, 1600, 900);
        imshow(myWindow, img_matches);
        preFrame = nowFrame;
        if(waitKey(delay) > 0)
            stop = true;
    }
    waitKey();
    return 0;
}

void mySift_function( int, void* ){
    if(myNfeatures > 0)
        myNfeatures = MAX(myNfeatures, 10);
    myOctaveLayers = MAX(myOctaveLayers, 2);
    mySigma = MAX(mySigma, 1);

    //cast
    dMyContrastThreshold =  static_cast<double>(myContrastThreshold)/100.0;
    dMyEdgeThreshold =  static_cast<double>(myEdgeThreshold)/100.0;
    dMySigma =  static_cast<double>(mySigma)/10.0;

    dMyRatihoThresh =  static_cast<double>(myRatioThresh)/100.0;
    cout << myNfeatures << " " << myOctaveLayers << " " << dMyContrastThreshold << " " << dMyEdgeThreshold << " " << dMySigma << endl;
}

