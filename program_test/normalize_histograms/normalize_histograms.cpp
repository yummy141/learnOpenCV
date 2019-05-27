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

Mat normalize_histograms(const Mat& h){
    /** note: the matlab program can return score values like 1.0000,
     * however, this function can only return 1.00000012e0.0
     */
    Mat res;
    int row = h.rows;
    float eps = 2.2204e-16;
    Mat weights(1, h.cols, CV_32F);
    cout << h.rows << endl;
    /* model.vocab.weights = log((size(model.index.histograms,2)+1) ./ (max(sum(model.index.histograms > 0,2),eps))); 
    *  matlab histogram 4000*75
    */
    /** compute idf weights
     */
    for(int i = 0; i < h.cols; i++)
    {
        float count_n = 0;
        for(int j = 0; j < h.rows; j++){
            // if(j < 5 && i < 5)
            //     cout << "value: " << h.at<float>(j, i) << endl;
            if(h.at<float>(j, i) > 0)
            {
                if(i == 0)
                    cout << j << " " << i << endl;
                 count_n += 1.0;
            }
               
        }
        count_n = max(count_n, eps);
        // if(i < 10)
        //     cout << count_n << endl;
        weights.at<float>(0, i) = log(static_cast<float>(row + 1) /
                                    count_n);
    }
    FileStorage fs_weights("idf_weights.yml", FileStorage::WRITE);
    fs_weights << "weights" << weights;
    fs_weights.release();
     // cout << weights(Range(0,1), Range(0,10)) << endl;

    /** weights and normalize histograms
     *  for t = 1:length(model.index.ids)
            h = model.index.histograms(:,t) .*  model.vocab.weights ;
            model.index.histograms(:,t) = h / norm(h) ;
        end
     */
    for(int i = 0; i < h.rows; i++)
    {
        Mat temp_h(1, h.cols, CV_32F);
        float temp_sqrt_sum2 = 0.0;
        for(int j = 0; j < h.cols; j++)
        {
            temp_h.at<float>(0, j) = weights.at<float>(0, j) * h.at<float>(i, j);
            temp_sqrt_sum2 += temp_h.at<float>(0, j) * temp_h.at<float>(0, j);
        }
        temp_sqrt_sum2 = sqrt(temp_sqrt_sum2);
//        cout << temp_h << endl;
//        cout << temp_sqrt_sum << endl;
        temp_h = temp_h / temp_sqrt_sum2;
        res.push_back(temp_h);
    }
    weights.release();
   // cout << res(Range(0,10), Range(0,10)) << endl;
    return res;
}

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
    
    FILE *f = fopen("/home/ian/coding_MATLAB/SUN3Dsfm-master/historgrams.in", "rb");
    float* buffer;
    size_t result;
    buffer = (float*)malloc(sizeof(float)*75*4000);
    result = fread(buffer, sizeof(float), 75*4000, f);

    fclose(f);

    Mat histogram(75, 4000, CV_32FC1, buffer);
    Mat res = normalize_histograms(histogram);
    
    FileStorage histogram_r("histogram.yml", FileStorage::WRITE);
    histogram_r << "histogram" << histogram;
    histogram_r.release();
    // cout << res(Range(0,10), Range(0,10)) << endl;
    Mat scores;
    scores = res * res.t();
    // cout << scores(Range(0,10), Range(0,10)) << endl;

    Mat scores_dst, temp_dst;
    Mat temp = Mat::ones(scores.rows, scores.cols, CV_8UC1);
    for(int i = 0; i < scores.rows; i++)
        temp.at<u_char>(i, i) = 0;
    // cout << temp;
    distanceTransform(temp, temp_dst, DIST_L2, DIST_MASK_PRECISE);
    cout << temp_dst(Range(0,10), Range(0,10)) << endl;
    cout << temp_dst.type() << endl;
    temp.release();

    for(int i = 0; i < scores.rows; i++)
        for(int j = 0; j < scores.cols; j++ )
        {
            if(j >= i)
                scores.at<float>(i, j) = 0.0;
            else{
                scores.at<float>(i, j) = min(temp_dst.at<float>(i, j) / 30.0, 1.0) * scores.at<float>(i, j);
            }
        }
 
    cout << scores(Range(0,10), Range(0,10)) << endl;

   /// cout << "scores ends. ";
 //    cout << scores;
 //    scores = scores_dst;
 
    GaussianBlur( scores, scores_dst, Size( 5, 5 ), 2 );


    Mat res_nonmaxup = nonmaxup(scores, 7);
    for(int i = 0; i < res_nonmaxup.rows; i++)
        for(int j = 0; j < res_nonmaxup.cols; j++)
            if(res_nonmaxup.at<float>(i, j) > 0)
                cout << i+1 << " " << j+1 << endl;

    // Mat temp_g = getGaussianKernel ( 5, 2, CV_32F ); 
    // Mat G = temp_g * temp_g.t();		
    // cout << G;
    // cout << histogram(Range(0,10), Range(0,10)) << endl;
    return 0;
}


