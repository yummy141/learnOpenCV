#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include <omp.h>
#include "opencv2/core.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
using namespace cv::cuda;
void my_surf_test(int i);
void my_gpu_surf_test(int i);

int n = 10;
std::vector< std::vector<KeyPoint> > resultKP(n + 1);
std::vector< std::vector<float> > resultDP(n + 1);

int main( int argc, char* argv[] )
{

    double t1 = omp_get_wtime( );

 #pragma omp parallel for
    for (int i = 0; i <= n; i++) {
        my_gpu_surf_test(i);
    }
    double t2 = omp_get_wtime( );

    cout << "parallel(MP + CUDA): " << t2 - t1 << endl;

    t1 = omp_get_wtime( );
    for (int i = 0; i <= n; i++) {
        my_gpu_surf_test(i);
    }
    t2 = omp_get_wtime( );

    cout << "parallel(CUDA): " << t2 - t1 << endl;

    t1 = omp_get_wtime( );
#pragma omp parallel for
    for (int i = 0; i <= n; i++) {
        my_surf_test(i);
    }
    t2 = omp_get_wtime( );

    cout << "parallel(MP): " << t2 - t1 << endl;

    for (int i = 0; i <= n; i++) {
        my_surf_test(i);
    }
    t2 = omp_get_wtime( );

    cout << "serial: " << t2 - t1 << endl;
    waitKey();
    return 0;
}

void my_surf_test(int i){

    char s[100];
    std::sprintf(s, "../data/color/%06d.jpg", i);
    std::string t(s);
    Mat src = imread(t, IMREAD_GRAYSCALE);
    // int minHessian = 400;

    Ptr<SURF> detector = SURF::create();
//    std::vector<KeyPoint> keypoints;
//    Mat descriptors;
    detector->detectAndCompute( src, noArray(), resultKP[i], resultDP[i] );

//    Mat img_keypoints;
//    drawKeypoints( src, resultKP[0], img_keypoints );
//    imshow("CPU-SURF Keypoints", img_keypoints );
}

void my_gpu_surf_test(int i){
    char s[100];
    std::sprintf(s, "../data/color/%06d.jpg", i);
    std::string t(s);
    GpuMat img1;

    Mat src = imread(t, IMREAD_GRAYSCALE);
    img1.upload(src);

    // cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice()); 输出设备信息

    SURF_CUDA surf;

    GpuMat keypointsGPU;
    GpuMat descriptorsGPU;

    surf(img1, GpuMat(), keypointsGPU, descriptorsGPU);

//    std::vector<KeyPoint> keypoints;
//    std::vector<float> descriptors;

    surf.downloadKeypoints(keypointsGPU, resultKP[i]);
    surf.downloadDescriptors(descriptorsGPU, resultDP[i]);

//    Mat img_keypoints;
//    drawKeypoints( src, resultKP[0], img_keypoints );
//    imshow("GPU-SURF Keypoints", img_keypoints );

}