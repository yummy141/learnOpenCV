 #include <opencv2/imgcodecs.hpp>
 #include <opencv2/xfeatures2d/cuda.hpp>
 #include <opencv2/cudafeatures2d.hpp>
 using namespace std;
 int GetMatchPointCount(string pic_path_1, string pic_path_2) {
     /*指定使用的GPU序号，相关的还有下面几个函数可以使用
 cv::cuda::getCudaEnabledDeviceCount();
 cv::cuda::getDevice();
 cv::cuda::DeviceInfo*/
     cv::cuda::setDevice(0);
     /*向显存加载两张图片。这里需要注意两个问题：
    第一，我们不能像操作（主）内存一样直接一个字节一个字节的操作显存，也不能直接从外存把图片加载到显存，一般需要通过内存作为媒介
    第二，目前opencv的GPU SURF仅支持8位单通道图像，所以加上参数IMREAD_GRAYSCALE*/
     cv::cuda::GpuMat gmat1;
     cv::cuda::GpuMat gmat2;
     gmat1.upload(cv::imread(pic_path_1,cv::IMREAD_GRAYSCALE));
     gmat2.upload(cv::imread(pic_path_2,cv::IMREAD_GRAYSCALE));
     /*下面这个函数的原型是：
  explicit SURF_CUDA(double
      _hessianThreshold, //SURF海森特征点阈值
      int _nOctaves=4, //尺度金字塔个数
      int _nOctaveLayers=2, //每一个尺度金字塔层数
      bool _extended=false, //如果true那么得到的描述子是128维，否则是64维
      float _keypointsRatio=0.01f,
      bool _upright = false
      );
  要理解这几个参数涉及SURF的原理*/
     cv::cuda::SURF_CUDA surf( 100,4,3 );
     /*分配下面几个GpuMat存储keypoint和相应的descriptor*/
     cv::cuda::GpuMat keypt1,keypt2; cv::cuda::GpuMat desc1,desc2;
     /*检测特征点*/
     surf(gmat1,cv::cuda::GpuMat(),keypt1,desc1);
     surf(gmat2,cv::cuda::GpuMat(),keypt2,desc2);
     /*匹配，下面的匹配部分和CPU的match没有太多区别,这里新建一个Brute-Force Matcher，一对descriptor的L2距离小于0.1则认为匹配*/
     auto matcher=cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
     vector<cv::DMatch> match_vec;
     matcher->match(desc1,desc2,match_vec);
     int count=0;
     for(auto & d:match_vec){
         if(d.distance<0.1)
             count++;
     }
     return count;
 }

 int main(int argc, const char* argv[]) {
     GetMatchPointCount("../data/color/000000.jpg", "../data/color/000002.jpg");
     return 0;
 }

