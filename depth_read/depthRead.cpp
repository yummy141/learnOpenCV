#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

Mat depthRead(const string filename){
    /** Note: In Matlab, bit computation(& 65535) will be casted since depth is typed as unsigned int 16
     *  However, this cannot be implemented automatically in C++, so i manually do this.
    */
    Mat depth;
	
    depth = imread(filename, CV_16UC1);
    // cout << (depth.at<short>(100, 100)) << endl;
    cout << depth(Range(100,110), Range(100,110)) << endl;
    Mat depth_res(depth.rows, depth.cols, CV_32F);
    cout << "depth." << endl;
    for(int i = 0; i < depth.rows; i++)
        for(int j = 0; j < depth.cols; j++){
            depth.at<unsigned short>(i, j) = ( ((depth.at<unsigned short>(i, j) >> 3) & 65535) | ((depth.at<unsigned short>(i, j) << 13) & 65535) );
            depth_res.at<float>(i, j) = depth.at<unsigned short>(i, j) / static_cast<float>(1000);
        }
   //  cout << "depth ends." << endl;
   cout << depth_res(Range(100,110), Range(100,110)) << endl;
    return depth;
}


int main( int argc, char* argv[] )
{
    string x = "/home/ian/CLionProjects/cuda/hotel_umd/maryland_hotel3/depth/0000001-000000000000.png";
    string y = "/home/ian/CLionProjects/cuda/hotel_umd/maryland_hotel3/my_depth/0000001-000000000000.png";
	
    Mat t = depthRead(x);
    Mat depth_i(480, 640, CV_32F);
    FileStorage fs_i(y, FileStorage::READ);
    fs_i["depth"] >> depth_i;
    cout << "test flag" << endl;
    cout << depth_i(Range(100,110), Range(100,110)) << endl;
    fs_i.release();
    
     cout << (((25664 >>3)&65535) | ((25664 << 13)&65535)) << endl;
     // cout << (25664 >>3)&65535);
    return 0;
}
