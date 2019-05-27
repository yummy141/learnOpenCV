#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
// #include <opencv2/ml/ml.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
//#include <Windows.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <thread>
// #include <omp.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstdio>
// #define NUM_THREADS 8
using namespace std;
using namespace cv;

Mat nonmaxup(Mat m, int radius){
    int sze = 2 * radius + 1;
    Mat ll(sze, sze, CV_32F);
    for(int i = 0; i < sze; i++)
        for(int j = 0; j < sze; j++)
            ll.at<float>(i, j) = static_cast<float>(j + 1);
    Mat rr = ll.t();

   //  cout << "correct: " << (cv::getStructuringElement ( MORPH_RECT, Size(15, 15), Point(-1, -1))).type() << endl;
    Mat areaM(sze, sze, CV_8U);
    // cout << areaM;
    for(int i = 0; i < sze; i++)
        for(int j = 0; j < sze; j++){
            auto temp1 = (ll.at<float>(i, j) - (radius + 1)) * (ll.at<float>(i, j) - (radius + 1));
            auto temp2 = (rr.at<float>(i, j) - (radius + 1)) * (rr.at<float>(i, j) - (radius + 1));
            if(sqrt(temp1 + temp2) < static_cast<float>(sze)/2.0)
                areaM.at<uchar>(i, j) = 1;
            else
                areaM.at<uchar>(i, j) = 0;
        }

    // cout << areaM << endl;
 //   cout << "false: " <<  areaM.type() << endl;
    // FileStorage fs4("areaM.yml", FileStorage::WRITE);
    // fs4 << "areaM" << areaM;
    // fs4.release();

    Mat mx;
    dilate(m, mx, areaM);
    cout << "areaM over" << endl;
    Mat selectMap = Mat::zeros(m.rows, m.cols, CV_32F);
    for(int i = 0; i < m.rows; i++)
        for(int j = 0; j < m.cols; j++)
            if(m.at<float>(i, j) == mx.at<float>(i, j))
                selectMap.at<float>(i, j) = 1.0;

    // dilate(selectMap, selectMap, Mat::ones(3,3,CV_32F));
    dilate(selectMap, selectMap, Mat::ones(3,3,CV_32F));
    // cout << selectMap;
    // FileStorage fs6("selectMap.yml", FileStorage::WRITE);
    // fs6 << "selectMap" << selectMap;
    // fs6.release();
    // cout << selectMap;
    // cout << m.type() << endl;
    for(int i = 0; i < m.rows; i++)
        for(int j = 0; j < m.cols; j++)
            if(selectMap.at<float>(i, j) != 1.0)   // m .* selectmap
                m.at<float>(i, j) =  0.0;

    cout << "over" << endl;
    // cout << m << endl;
    // FileStorage fs5("m.yml", FileStorage::WRITE);
    // fs5 << "m" << m;
    // fs5.release();

    ll.release();
    rr.release();
    selectMap.release();
    mx.release();
    areaM.release();

    return m;

}

int main(int argc, char **argv){
    
    FILE *f = fopen("/home/ian/coding_MATLAB/SUN3Dsfm-master/scores.in", "rb");
    float* buffer;
    size_t result;
    buffer = (float*)malloc(sizeof(float)*75*75);
    result = fread(buffer, sizeof(float), 75*75, f);

    fclose(f);

    Mat scores(75, 75, CV_32FC1, buffer);
    cout << scores(Range(0,10), Range(0,10)) << endl;
    Mat res = nonmaxup(scores, 7);
    for(int i = 0; i < res.rows; i++)
        for(int j = 0; j < res.cols; j++)
            if(res.at<float>(i, j) > 0)
                cout << i+1 << " " << j+1 << endl;
    // cout << res(Range(10,15), Range(0, 5)) << endl;
    // cout << res << endl;
    // Mat temp_g = getGaussianKernel ( 5, 2, CV_32F ); 
    // Mat G = temp_g * temp_g.t();		
    // cout << G;
    // cout << histogram(Range(0,10), Range(0,10)) << endl;
    return 0;
}


