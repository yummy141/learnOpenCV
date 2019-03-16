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
Mat img_object, img_scene;

// for sift
int myNfeatures = 0, maxNfeatures = 200;
int myOctaveLayers = 3, maxOctaveLayers = 20;
int myContrastThreshold = 4, maxContrastThreshold = 30;
int myEdgeThreshold = 10, maxEdgeThreshold = 100;
int mySigma = 16, maxSigma = 50;

// for window
const char* mySift_window = "This is sift test";
void mySift_function( int, void* );

// for match
int myRatioThresh = 75, maxRatioThresh = 100;

int main( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, keys );
    img_object = imread( parser.get<String>("input1"), IMREAD_GRAYSCALE );
    img_scene = imread( parser.get<String>("input2"), IMREAD_GRAYSCALE );
    if ( img_object.empty() || img_scene.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }


    namedWindow( mySift_window );
    createTrackbar( "nfeatures:", mySift_window, &myNfeatures, maxNfeatures, mySift_function );
    createTrackbar( "nOctaveLayers:", mySift_window, &myOctaveLayers, maxOctaveLayers, mySift_function );
    createTrackbar( "contrastThreshold(*100):", mySift_window, &myContrastThreshold, maxContrastThreshold, mySift_function );
    createTrackbar( "edgeThreshold(*100) :", mySift_window, &myEdgeThreshold, maxEdgeThreshold, mySift_function );
    createTrackbar( "sigma(*10):", mySift_window, &mySigma, maxSigma, mySift_function );
    createTrackbar( "ratio_thresh(*100):", mySift_window, &myRatioThresh, maxRatioThresh, mySift_function );
    mySift_function( 0, 0 );

    waitKey();
    return 0;
}

void mySift_function( int, void* ) {
    // change parameters according to the Trackbar
    if(myNfeatures > 0)
        myNfeatures = MAX(myNfeatures, 10);
    myOctaveLayers = MAX(myOctaveLayers, 2);
    mySigma = MAX(mySigma, 1);

    //cast
    double dMyContrastThreshold =  static_cast<double>(myContrastThreshold)/100.0;
    double dMyEdgeThreshold =  static_cast<double>(myEdgeThreshold)/100.0;
    double dMySigma =  static_cast<double>(mySigma)/10.0;
    cout << myNfeatures << " " << myOctaveLayers << " " << dMyContrastThreshold << " " << dMyEdgeThreshold << " " << dMySigma << endl;

    //-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
    Ptr<SIFT> detector = SIFT::create(myNfeatures, myOctaveLayers, dMyContrastThreshold, dMyEdgeThreshold, dMySigma);
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test

    double dMyRatihoThresh =  static_cast<double>(myRatioThresh)/100.0;

    // const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
        if (knn_matches[i][0].distance < dMyRatihoThresh * knn_matches[i][1].distance)
            good_matches.push_back(knn_matches[i][0]);

    //-- Draw matches
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

//     //Homography
//    Mat H = findHomography( obj, scene, RANSAC );
//
//    //-- Get the corners from the image_1 ( the object to be "detected" )
//    std::vector<Point2f> obj_corners(4);
//    obj_corners[0] = Point2f(0, 0);
//    obj_corners[1] = Point2f( (float)img_object.cols, 0 );
//    obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
//    obj_corners[3] = Point2f( 0, (float)img_object.rows );
//    std::vector<Point2f> scene_corners(4);
//    perspectiveTransform( obj_corners, scene_corners, H);
//
//    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
//    line( img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
//          scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4 );
//    line( img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
//          scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
//          scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
//          scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//
//    //-- Show detected matches
    imshow(mySift_window, img_matches );
}
