#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <iostream>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
Mat src, src_gray;
Mat mySift_copy;

int myNfeatures = 0, maxNfeatures = 200;
int myOctaveLayers = 3, maxOctaveLayers = 20;
int  myContrastThreshold = 4, maxContrastThreshold = 100;
int  myEdgeThreshold = 10, maxEdgeThreshold = 100;
int  mySigma = 16, maxSigma = 50;
const char* mySift_window = "This is sift test";
void mySift_function( int, void* );
int main( int argc, char** argv )
{

    // input
    CommandLineParser parser( argc, argv, "{@input | ../data/building.jpg | input image}" );
    src = imread( parser.get<String>( "@input" ) );
    if ( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    namedWindow( mySift_window );
    createTrackbar( "nfeatures:", mySift_window, &myNfeatures, maxNfeatures, mySift_function );
    createTrackbar( "nOctaveLayers:", mySift_window, &myOctaveLayers, maxOctaveLayers, mySift_function );
    createTrackbar( "contrastThreshold(*100):", mySift_window, &myContrastThreshold, maxContrastThreshold, mySift_function );
    createTrackbar( "edgeThreshold(*100) :", mySift_window, &myEdgeThreshold, maxEdgeThreshold, mySift_function );
    createTrackbar( "sigma(*10):", mySift_window, &mySigma, maxSigma, mySift_function );


    mySift_function( 0, 0 );

    waitKey();
    return 0;
}

void mySift_function( int, void* ) {
    mySift_copy = src.clone();

    // change parameters according to the Trackbar
    myOctaveLayers = MAX(myOctaveLayers, 1);
    // myNfeatures = MAX(myNfeatures, 1)
    mySigma = MAX(mySigma, 1);

    // cast
    double dMyContrastThreshold =  static_cast<double>(myContrastThreshold)/100.0;
    double dMyEdgeThreshold =  static_cast<double>(myEdgeThreshold)/100.0;
    double dMySigma =  static_cast<double>(mySigma)/10.0;
    cout << myNfeatures << " " << myOctaveLayers << " " << dMyContrastThreshold << " " << dMyEdgeThreshold << " " << dMySigma << endl;
    // create detector
    Ptr<SIFT> detector = SIFT::create(myNfeatures, myOctaveLayers, dMyContrastThreshold, dMyEdgeThreshold, dMySigma);

    // detect
    std::vector<KeyPoint> keypoints;
    Mat img_keypoints;
    detector->detect(mySift_copy, keypoints);

    // draw
    drawKeypoints( mySift_copy, keypoints, img_keypoints );
    imshow(mySift_window, img_keypoints);
}