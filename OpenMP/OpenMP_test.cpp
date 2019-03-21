#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <time.h>
#include <omp.h>
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
using std::string;

void my_sift_test(int i);

int main( int argc, char* argv[] )
{
    // CommandLineParser parser( argc, argv, "{@input | ../data/box.png | input image}" );
    double t1 = omp_get_wtime( );
#pragma omp parallel for
    for (int i = 0; i <= 944; i++) {
        my_sift_test(i);
    }

    double t2 = omp_get_wtime( );

    double t3 = omp_get_wtime( );
    for (int i = 0; i <= 944; i++) {
        my_sift_test(i);
    }

    double t4 = omp_get_wtime( );

    cout << "parallel: " << t2 - t1 << endl;
    cout << "serial: " << t4 - t3 << endl;
    return 0;
}

void my_sift_test(int i){
    char s[100];
    std::sprintf(s, "../data/color/%06d.jpg", i);
    string t(s);
    // t = "../data/color/" + t + ".jpg";
//    cout << i << " " << t << endl;
    Mat src = imread(t, IMREAD_GRAYSCALE);
    // imshow(t, src);
    // waitKey();
    if (src.empty()) {
        cout << "Could not open or find the image!\n" << endl;
        cout << i << endl;
        // return -1;
    }
    Ptr<SIFT> detector = SIFT::create();
    SiftDescriptorExtractor extractor;
    std::vector<KeyPoint> kp;
    Mat descriptors;
    detector->detectAndCompute(src, noArray(), kp, descriptors);
}
