
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>
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

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h> // pclNormalEstimation
#include <pcl/registration/ndt.h> // ndt
#include <pcl/filters/approximate_voxel_grid.h>


#include <thread>

#include <opencv2/core/eigen.hpp> //cv2eigen



using namespace std;
using namespace cv;
//using namespace cv::ml;
vector<string> category;
vector<string> get_categories(const string& dir){
    DIR *dp;
    struct dirent *dirp;
    struct stat filestat;
    string filepath;

    dp = opendir( dir.c_str() );
    vector<string> res;
    while(dirp = readdir( dp )){

        filepath = dir + "/" + dirp->d_name;
        // If the file is a directory (or is in some way invalid) we'll skip it
        if (stat( filepath.c_str(), &filestat ) || S_ISDIR( filestat.st_mode )){
            if(static_cast<string>(dirp->d_name) == "." || static_cast<string>(dirp->d_name) == "..")
                continue;
            else{
                res.emplace_back(static_cast<string>(dirp->d_name));
                // res.emplace_back(filepath);
                // cout << filepath << endl;
            }

        }
        // cout << filepath << endl;
    }
    return res;
}

Mat allDescriptors;

void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
{
    /***************************************
    * xgv -- 【输入】指定X输入范围
    * ygv -- 【输入】指定Y输入范围
    * X   -- 【输出】Mat
    * Y   -- 【输出】Mat
    * usage: meshgrid(Range(1, 5), Range(1,6), X, Y);
    ****************************************/

    std::vector<float> t_x, t_y;
    for(int i = xgv.start; i <= xgv.end; i++) t_x.push_back(static_cast<float>(i));
    for(int j = ygv.start; j <= ygv.end; j++) t_y.push_back(static_cast<float>(j));
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X); // repeat along the vertical axis
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y); // repeat along the horizontal axis
}


void DetectComputeimage(const string& folder_path) {

    DIR *dp;
    struct dirent *dirp;
    struct stat filestat;
    string filepath;

    dp = opendir( folder_path.c_str() );
    // vector<string> res;
    while (dirp = readdir( dp )){
        filepath = folder_path + "/" + dirp->d_name;
        // If the file is a directory (or is in some way invalid) we'll skip it
        if (stat( filepath.c_str(), &filestat ) || S_ISDIR( filestat.st_mode )){
                continue;
            }

        if (filepath.find(".jpg") != -1){
            cout <<"processing: " << filepath << endl;

            Ptr<xfeatures2d::SIFT> siftptr;
            siftptr = xfeatures2d::SIFT::create();
            Mat img;
            img = imread(filepath);
            // Mat img_gray;
            // cvtColor(img, img_gray, COLOR_BGR2GRAY);

            vector<KeyPoint> keypoints;
            Mat descriptors;
            siftptr->detectAndCompute(img, noArray(), keypoints, descriptors);

            cout << "descriptors size: " << descriptors.size << endl;
            allDescriptors.push_back(descriptors);
           //  allDescPerImg.emplace_back(descriptors);
        }
    }
    return;

//    allDescPerImgNum++;
}

Mat create_codebook(){

    auto number_of_descriptors = static_cast<float>(allDescriptors.rows);
    // int number_of_words = 1000;
    auto number_of_words = static_cast<int>(sqrt(number_of_descriptors));

    number_of_words = 3; // for test


    cout << "Total descriptors: " << number_of_descriptors << endl;

    FileStorage fs("training_descriptors.yml", FileStorage::WRITE);
    fs << "training_descriptors" << allDescriptors;
    fs.release();

    cout << "Now descriptors: " << number_of_words << endl;
    BOWKMeansTrainer bowtrainer(number_of_words, TermCriteria(), 1); //num clusters
    // 默认attemp = 3
    bowtrainer.add(allDescriptors);
    cout << "cluster BOW features..." << endl;
    Mat vocabulary = bowtrainer.cluster();

    FileStorage fs1("vocabulary.yml", FileStorage::WRITE);
    fs1 << "vocabulary" << vocabulary;
    fs1.release();

    return vocabulary;
}

map<int, string> imagei2s;      // matrixID
map<int, string> imagei2ds;
map<int, string> IDimage2depth; // frameID, 0,1,2,3...
map<int, string> ID2image;
Mat histograms;
Mat train_label;

struct pairs{
//    Mat R;
//    Mat t;
    Eigen::Matrix<double, 3, 3> R;
    Eigen::Matrix<double, 3, 1> t;
    int i;
    int j;
    vector<Point> valid_i;
    vector<Point> valid_j;
    vector<vector<float>> P3D_i;
    vector<vector<float>> P3D_j;
};

void create_bow_histogram_features(Mat codebook, string folder_path){
    Ptr<FeatureDetector > detector = xfeatures2d::SIFT::create(); //detector
    Ptr<DescriptorExtractor > extractor = xfeatures2d::SIFT::create();//  extractor;
//    Ptr<DescriptorExtractor > extractor(
//            new OpponentColorDescriptorExtractor(
//                    Ptr<DescriptorExtractor>(new SurfDescriptorExtractor())
//            )
//    );
    Ptr<DescriptorMatcher > matcher(new BFMatcher());
    BOWImgDescriptorExtractor bowide(extractor, matcher);
    bowide.setVocabulary(codebook);

    DIR *dp;
    struct dirent *dirp;
    struct stat filestat;
    string filepath;
    dp = opendir( folder_path.c_str() );

    int file_num = 0;
    while (dirp = readdir( dp )){
        filepath = folder_path + "/" + dirp->d_name;
        // If the file is a directory (or is in some way invalid) we'll skip it
        if (stat( filepath.c_str(), &filestat ) || S_ISDIR( filestat.st_mode )){
            continue;
        }

        if (filepath.find(".jpg") != -1){
            cout <<"processing: " << filepath << endl;

            Mat img = imread(filepath);
            //  Mat grayimg;
            //  cvtColor(imread(img, grayimg, CV_BGR2GRAY);
            Mat response_hist;

            /** setting image file & depth map
             *  Notice: FramID starts from 1
             */
            imagei2s[file_num] = filepath;

            int FrameID, _Timestamp;
            sscanf(dirp->d_name, "%d-%d.png", &FrameID, &_Timestamp);
            ID2image[FrameID - 1] = filepath;


            imagei2ds[file_num] = IDimage2depth[FrameID - 1];

            file_num++;
            /** detect keypoints and compute histograms
             */
            vector<KeyPoint> keypoints;
            detector->detect(img, keypoints);
            bowide.compute(img, keypoints, response_hist);
            //  cout << "response_hist size: " << response_hist.size() << endl;

           /** the response_hist is somehow divided by keypoints.size(),
            * which i don't expect. So I multiplied the result with keypoints.size()
            */
            response_hist = response_hist * keypoints.size();

            //  cout << response_hist << endl;
            //  classes_training_data[filepath] = response_hist;
            histograms.push_back(response_hist);
        }
    }
    return;
}

Mat normalize_histograms(const Mat& h){
    /** note: the matlab program can return score values like 1.0000,
     * however, this function can only return 1.00000012e0.0
     */
    Mat res;
    int row = h.rows;
    float eps = 2.2204e-16;
    Mat weights(1, h.cols, CV_32F);
    /** compute idf weights
     */
    for(int i = 0; i < h.cols; i++)
    {
        float count_n = 0;
        for(int j = 0; j < h.rows; j++)
            if(h.at<float>(j, i) > 0)
                count_n += 1.0;
        count_n = max(count_n, eps);
        weights.at<float>(0, i) = log(static_cast<float>(row + 1) /
                                    count_n);
    }
    FileStorage fs_weights("weights.yml", FileStorage::WRITE);
    fs_weights << "weights" << weights;
    fs_weights.release();

    /** weights and normalize histograms
     */
    for(int i = 0; i < h.rows; i++)
    {
        Mat temp_h(1, h.cols, CV_32F);
        double temp_sqrt_sum2 = 0.0;
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
    return res;
}

void create_depth_map(const string folder_path){
    DIR *dp;
    struct dirent *dirp;
    struct stat filestat;
    string filepath;
    dp = opendir( folder_path.c_str() );

    int file_num = 0;
    while (dirp = readdir( dp )) {
        filepath = folder_path + "/" + dirp->d_name;
        /** If the file is a directory (or is in some way invalid) we'll skip it
         */
        if (stat(filepath.c_str(), &filestat) || S_ISDIR(filestat.st_mode)) {
            continue;
        }
        int FrameID, Timestamp;
        sscanf(dirp->d_name, "%d-%d.png", &FrameID, &Timestamp);

        /** note: the filename startes from index 1, but we expect index starts from 0
         */
        IDimage2depth[FrameID - 1] = filepath;
        file_num++;
    }

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
//    FileStorage fs4("areaM.yml", FileStorage::WRITE);
//    fs4 << "areaM" << areaM;
//    fs4.release();

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
    FileStorage fs6("selectMap.yml", FileStorage::WRITE);
    fs6 << "selectMap" << selectMap;
    fs6.release();
    // cout << selectMap;
    // cout << m.type() << endl;
    for(int i = 0; i < m.rows; i++)
        for(int j = 0; j < m.cols; j++)
            if(selectMap.at<float>(i, j) != 1.0)   // m .* selectmap
                m.at<float>(i, j) =  0.0;

    cout << "over" << endl;
    // cout << m << endl;
    FileStorage fs5("m.yml", FileStorage::WRITE);
    fs5 << "m" << m;
    fs5.release();

    ll.release();
    rr.release();
    selectMap.release();
    mx.release();
    areaM.release();

    return m;

}

Mat readK(string dir_k){
    Mat K(3, 3, CV_32F);
    ifstream fk;
    fk.open(dir_k);
    // stringstream ss;
    string temp_s;
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++){
            fk >> temp_s;
            K.at<float>(i, j) = atof(temp_s.c_str());
        }
    fk.close();
    return K;
}


/* Wrong function due to convert_to */
Mat depthRead(const string filename){
    /** Note: In Matlab, bit computation(& 65535) will be casted since depth is typed as unsigned int 16
     *  However, this cannot be implemented automatically in C++, so i manually do this.
    */
    Mat depth;

    depth = imread(filename, CV_16UC1);

    cout << "depth." << endl;
    for(int i = 0; i < depth.rows; i++)
        for(int j = 0; j < depth.cols; j++){
            // cout << depth.at<int>(i, j) << endl;
            depth.at<int>(i, j) = ( (((depth.at<int>(i, j) & 65535) >> 3) & 65535) | (((depth.at<int>(i, j) & 65535) << 13) & 65535) ) & 65535;
            // depth_res.at<double>(i, j) = depth.at<int>(i, j) / static_cast<double>(1000);
            // cout << depth_res.at<double>(i, j) << endl;
        }
    cout << "depth ends." << endl;
//    Mat depth_res(depth.rows, depth.cols, CV_32F);
//    depth.convertTo(depth_res, CV_32F);
//    depth_res /= 1000.0;
//    depth.release();
//    FileStorage depthW("depth.yml", FileStorage::WRITE);
//    depthW << "depth" << depth;
//    depthW.release();
    // depth.convertTo(depth_res, CV_32F);
    // cout << depth_res;
    // depth_res /= 1.0;
    // cout << depth_res;
    // cout << "depth_res: " << endl;
    return depth;
}

Mat convert_depth(const Mat depth){
//    Mat depth_res(depth.rows, depth.cols, CV_32F);
    Mat depth_res;
    cout<<"flag"<<endl;
    depth.convertTo(depth_res, CV_32FC1);
    depth_res /= 1000.0;

    return depth_res;
}

pcl::visualization::PCLVisualizer::Ptr simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}

void rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        std::this_thread::sleep_for(100ms);
    }
}

vector<Mat> depth2XYZcamera(const Mat& depth , const Mat& K){
    /** compute xyz CAMERA from depth  OpenCVMat
     */
    Mat X,Y;
    meshgrid(Range(1, 640), Range(1, 480), X, Y); // 480 * 640

    /** Notice: index should start from zero */
    Mat XYZcamera_i1 = (X - K.at<float>(0,2)).mul(depth) / K.at<float>(0,0);
    Mat XYZcamera_i2 = (Y - K.at<float>(1,2)).mul(depth) / K.at<float>(1,1);
    Mat XYZcamera_i3 = depth;
    Mat XYZcamera_i4(XYZcamera_i3.rows, XYZcamera_i3.cols, CV_8UC1);
    for(int i = 0; i < XYZcamera_i3.rows; i++)
        for(int j = 0; j < XYZcamera_i3.cols; j++){
            if (depth.at<float>(i, j) == 0)
                XYZcamera_i4.at<uchar>(i, j) = 0;
            else
                XYZcamera_i4.at<uchar>(i, j) = 1;
    }
    vector<Mat> res;
    res.emplace_back(XYZcamera_i1);
    res.emplace_back(XYZcamera_i2);
    res.emplace_back(XYZcamera_i3);
    res.emplace_back(XYZcamera_i4);
    return res;
}

pairs* align2viewTimebased(int index_i, int index_j, const Mat& K){
    /**
     * index_i, index_j : filenumber
     * K : Matrix
     * return
     *   pair.Rt = RtRANSAC;
     *   pair.matches = [SIFTloc_i([2 1],:);P3D_i;SIFTloc_j([2 1],:);P3D_j];
     *   pair.i = frameID_i;
     *   pair.j = frameID_j;
     */
    float error3D_threshold = 0.05;
    float error3D_threshold2 = error3D_threshold * error3D_threshold;

    /** load two images and depth
     */
//    string test_A = "/home/ian/CLionProjects/cuda/hotel_umd/maryland_hotel3/depth/0000010-000000302323.png";
//    string test_B = "/home/ian/CLionProjects/cuda/hotel_umd/maryland_hotel3/depth/0000001-000000000000.png";
//    Mat depth_i = depthRead(test_A);
//    Mat depth_i_new = convert_depth(depth_i);
//    Mat depth_j = depthRead(test_B);
//    Mat depth_j_new = convert_depth(depth_j);

/*
    string t_s = "/home/ian/CLionProjects/cuda/hotel_umd/maryland_hotel3/my_depth/0000842-000028188444.png";
    FileStorage fs_test(t_s, FileStorage::READ);
    Mat t;
    fs_test["depth"] >> t;
    cout << t << endl;
*/

//    pairs test;
//    test.i = index_i;
//    test.j = index_j;

    pairs *P = new pairs;
    P->i = index_i;
    P->j = index_j;

    Mat image_i = imread(ID2image[index_i]);
    Mat image_j = imread(ID2image[index_j]);

//    cout << "image i: " << imagei2s[index_i] << "depth i: " << imagei2ds[index_i] << endl;
//    cout << "image j: " << imagei2s[index_j] << "depth j: " << imagei2ds[index_j] << endl;
    cout << "depth i: " << IDimage2depth[index_i] << "image i: " << ID2image[index_i] << endl;
    cout << "depth j: " << IDimage2depth[index_j] << "image j: " << ID2image[index_j] << endl;


    FileStorage fs_i(IDimage2depth[index_i], FileStorage::READ);
    FileStorage fs_j(IDimage2depth[index_j], FileStorage::READ);

    /** read depth, and the depth images are computed by MATLAB */
    Mat depth_i(480, 640, CV_32F);
    Mat depth_j(480, 640, CV_32F);
    fs_i["depth"] >> depth_i;
    fs_j["depth"] >> depth_j;

    fs_i.release();
    fs_j.release();

    /** Notice: index should start from zero */
    vector<Mat> XYZcamera_i = depth2XYZcamera(depth_i, K);
    vector<Mat> XYZcamera_j = depth2XYZcamera(depth_j, K);

//     FileStorage fs_XYZ_test("XYZ_test.yml", FileStorage::WRITE);
//     fs_XYZ_test << "fs_XYZ_test_x" << XYZcamera_i[0];
// //    fs_XYZ_test << "fs_XYZ_test[1]" << XYZcamera_i[1];
// //    fs_XYZ_test << "fs_XYZ_test[2]" << XYZcamera_i[2];
//     fs_XYZ_test.release();

    /** detect sift keypoints location and descriptors
     */
    // cout << image_i << endl;
    Ptr<xfeatures2d::SIFT>siftptr = xfeatures2d::SIFT::create(0, 4, 0.035, 10, 1.8);
    vector<KeyPoint> keypoints_i, keypoints_j;
    Mat descriptors_i, descriptors_j;
    siftptr->detectAndCompute(image_i, noArray(), keypoints_i, descriptors_i);
    siftptr->detectAndCompute(image_j, noArray(), keypoints_j, descriptors_j);

    cout << "Sift Keypoints size: " << keypoints_i.size() << " " << keypoints_j.size() << endl;

    /** sift matching
     */
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors_i, descriptors_j, knn_matches, 2 );

    float dMyRatihoThresh =  0.618f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
        if (knn_matches[i][0].distance < dMyRatihoThresh * knn_matches[i][1].distance)
            good_matches.push_back(knn_matches[i][0]);

//    /* plot Matches */
//    Mat img_matches;
//    drawMatches( image_i, keypoints_i, image_j, keypoints_j, good_matches, img_matches, Scalar::all(-1),
//                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//    const string testS2 = "Test";
//    imshow(testS2, img_matches);
//    waitKey(0);

    // cout << "Question is under this line 524" << endl;
    /** find keypoints_match
    */
    vector<KeyPoint> keypoints_match_i, keypoints_match_j;
    for(auto match : good_matches){
        // cout << match.queryIdx << " " << match.trainIdx << " "<< endl;
        if(match.queryIdx >= 0 && match.queryIdx < keypoints_i.size()
           && match.trainIdx >= 0 && match.trainIdx < keypoints_j.size())
        {
            keypoints_match_i.push_back(keypoints_i[match.queryIdx]);
            keypoints_match_j.push_back(keypoints_j[match.trainIdx]);
            // cout << "Keypoints: " << keypoints_i[match.queryIdx].pt.x << " " << keypoints_i[match.queryIdx].pt.y << endl;
        }
        else
            cout << "wrong match" << endl;
    }
    // cout << "Question is under this line 533" << endl;
    /** use set to delete duplicate points */
    set<tuple<int, int, int, int>> valid_keypoints;
    int temp_i_x, temp_i_y, temp_j_x, temp_j_y;
    for(int i = 0; i < keypoints_match_i.size(); i++){
        // cout << keypoints_match_i[i].pt.x << " " << keypoints_match_i[i].pt.y << " " << keypoints_match_j[i].pt.x << " " << keypoints_match_j[i].pt.y << endl;
        temp_i_x = static_cast<int>(round(keypoints_match_i[i].pt.y));
        temp_i_y = static_cast<int>(round(keypoints_match_i[i].pt.x));
        temp_j_x = static_cast<int>(round(keypoints_match_j[i].pt.y));
        temp_j_y = static_cast<int>(round(keypoints_match_j[i].pt.x));
        if(temp_i_x >= 0 && temp_i_x < depth_i.rows && temp_i_y >=0 && temp_i_y < depth_i.cols
           && temp_j_x >= 0 && temp_j_x < depth_j.rows && temp_j_y >= 0 && temp_j_y < depth_j.cols
           && depth_i.at<float>(temp_i_x, temp_i_y) != 0.0 && depth_j.at<float>(temp_j_x, temp_j_y) != 0.0)
        {
            valid_keypoints.insert(make_tuple(temp_i_x, temp_i_y, temp_j_x, temp_j_y));
//            cout << "Keypoints: " << temp_i_x << " " << temp_i_y << endl;
        }
    }
    // cout << "number of key points match: " << valid_keypoints.size() << endl;

    vector<Eigen::MatrixXd> XYZcamera_i_eigen(4);
    vector<Eigen::MatrixXd> XYZcamera_j_eigen(4);

    Eigen::MatrixXd XYZcamera_i_eigen_full(480*640, 3);
    Eigen::MatrixXd XYZcamera_j_eigen_full(480*640, 3);
    for(int i = 0; i < 4; i++){
        int row = XYZcamera_i[i].rows;
        int col = XYZcamera_i[i].cols;
        Eigen::MatrixXd temp_eigen_i(row, col);
        Eigen::MatrixXd temp_eigen_j(row, col);
        cv2eigen(XYZcamera_i[i], temp_eigen_i);
        cv2eigen(XYZcamera_j[i], temp_eigen_j);

        XYZcamera_i_eigen[i] = temp_eigen_i;
        XYZcamera_i_eigen[i].resize(480*640, 1);

        XYZcamera_j_eigen[i] = temp_eigen_j;
        XYZcamera_j_eigen[i].resize(480*640, 1);
    }

    XYZcamera_i_eigen_full << XYZcamera_i_eigen[0],
                                XYZcamera_i_eigen[1],
                                XYZcamera_i_eigen[2];

    XYZcamera_j_eigen_full << XYZcamera_j_eigen[0],
                                XYZcamera_j_eigen[1],
                                XYZcamera_j_eigen[2];


    vector<int> isValid_i, isValid_j;
    for(int i = 0; i < XYZcamera_i_eigen[3].size(); i++){
        if(XYZcamera_i_eigen[3](i) > 0)
            isValid_i.emplace_back(i);
        if(XYZcamera_j_eigen[3](i) > 0)
            isValid_j.emplace_back(i);
    }
    cout << "dense Valid size: " << isValid_i.size() << " " << isValid_j.size() << endl;
    /** dense */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_dense (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out_dense (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_rgb_dense (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out_rgb_dense (new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud_in_dense->height = 1; cloud_in_dense->is_dense = false;
    cloud_out_dense->height = 1; cloud_out_dense->is_dense = false;
    cloud_in_rgb_dense->height = 1; cloud_in_rgb_dense->is_dense = false;
    cloud_out_rgb_dense->height = 1; cloud_out_rgb_dense->is_dense = false;

    cloud_in_dense->width = isValid_i.size(); cloud_in_dense->points.resize (cloud_in_dense->width * cloud_in_dense->height);
    cloud_out_dense->width = isValid_j.size(); cloud_out_dense->points.resize (cloud_out_dense->width * cloud_out_dense->height);
    cloud_in_rgb_dense->width = isValid_i.size(); cloud_in_rgb_dense->points.resize (cloud_in_dense->width * cloud_in_dense->height);
    cloud_out_rgb_dense->width = isValid_j.size(); cloud_out_rgb_dense->points.resize (cloud_out_dense->width * cloud_out_dense->height);

    // cout << XYZcamera_i_eigen_full.rows() << " " << XYZcamera_i_eigen_full.cols() << endl;
    // cout << XYZcamera_j_eigen_full.rows() << " " << XYZcamera_j_eigen_full.cols() << endl;
    // random_shuffle(isValid_i.begin(), isValid_i.end());
    // random_shuffle(isValid_j.begin(), isValid_j.end());

    for(int i = 0; i < isValid_i.size(); i++){
        cloud_in_dense->points[i].x = XYZcamera_i_eigen_full(isValid_i[i], 0);
        cloud_in_rgb_dense->points[i].x = cloud_in_dense->points[i].x;
        cloud_in_dense->points[i].y = XYZcamera_i_eigen_full(isValid_i[i], 1);
        cloud_in_rgb_dense->points[i].y = cloud_in_dense->points[i].y;
        cloud_in_dense->points[i].z = XYZcamera_i_eigen_full(isValid_i[i], 2);
        cloud_in_rgb_dense->points[i].z = cloud_in_dense->points[i].z;
    }

    for(int i = 0; i < isValid_j.size(); i++){
        cloud_out_dense->points[i].x = XYZcamera_j_eigen_full(isValid_j[i], 0);
        cloud_out_rgb_dense->points[i].x = cloud_out_dense->points[i].x;
        cloud_out_dense->points[i].y = XYZcamera_j_eigen_full(isValid_j[i], 1);
        cloud_out_rgb_dense->points[i].y = cloud_out_dense->points[i].y;
        cloud_out_dense->points[i].z = XYZcamera_j_eigen_full(isValid_j[i], 2);
        cloud_out_rgb_dense->points[i].z = cloud_out_dense->points[i].z;
    }

//    for(int index = 0; index < isValid_i.size(); index++){
//
//        int i = isValid_i[index] % image_i.rows;
//        int j = isValid_i[index] / image_i.rows;
//        uint8_t r4(image_i.at<Vec3b>(i, j)[2]),
//                g4(image_i.at<Vec3b>(i, j)[1]),
//                b4(image_i.at<Vec3b>(i, j)[0]);
//        uint32_t rgb_temp1 = (static_cast<uint32_t>(r4) << 16 |
//                static_cast<uint32_t>(g4) << 8 | static_cast<uint32_t>(b4));
//        cloud_in_rgb_dense->points[index].rgb = *reinterpret_cast<float*>(&rgb_temp1);
//    }
//
//    for(int index = 0; index < isValid_j.size(); index++){
//
//        int i = isValid_j[index] % image_j.rows;
//        int j = isValid_j[index] / image_j.rows;
//        uint8_t r4(image_j.at<Vec3b>(i, j)[2]),
//                g4(image_j.at<Vec3b>(i, j)[1]),
//                b4(image_j.at<Vec3b>(i, j)[0]);
//        uint32_t rgb_temp2 = (static_cast<uint32_t>(r4) << 16 |
//                static_cast<uint32_t>(g4) << 8 | static_cast<uint32_t>(b4));
//        cloud_out_rgb_dense->points[index].rgb = *reinterpret_cast<float*>(&rgb_temp2);
//    }
//
//   //  rgbVis(cloud_in_rgb_dense);
//   //  rgbVis(cloud_out_rgb_dense);
//    pcl::io::savePCDFileASCII ("cloud_in_rgb.pcd", *cloud_in_rgb_dense);
//    pcl::io::savePCDFileASCII ("cloud_out_rgb.pcd", *cloud_out_rgb_dense);

    /** use matched kepoints to compute R and t
    */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);

    size_t index = 0;
    cloud_in->height   = 1; cloud_in->is_dense = false;
    cloud_in_rgb->height = 1; cloud_in->is_dense = false; 
    cloud_out->height   = 1; cloud_out_rgb->is_dense = false;
    cloud_out_rgb->height = 1; cloud_out->is_dense = false;

    /** color */
    uint8_t r1(255), g1(15), b1(15);
    uint8_t r2(15), g2(255), b2(15);
    uint8_t r3(255), g3(255), b3(15);

    uint32_t rgb1 = (static_cast<uint32_t>(r1) << 16 |
                     static_cast<uint32_t>(g1) << 8 | static_cast<uint32_t>(b1));
    uint32_t rgb2 = (static_cast<uint32_t>(r2) << 16 |
                     static_cast<uint32_t>(g2) << 8 | static_cast<uint32_t>(b2));
    uint32_t rgb3 = (static_cast<uint32_t>(r3) << 16 |
                     static_cast<uint32_t>(g3) << 8 | static_cast<uint32_t>(b3));



    for(auto coordinate : valid_keypoints){
        unsigned int num_of_keypoints = valid_keypoints.size();
        cloud_in->width = num_of_keypoints;
        cloud_in_rgb->width = num_of_keypoints;
        cloud_in->points.resize (cloud_in->width * cloud_in->height);
        cloud_in_rgb->points.resize (cloud_in->width * cloud_in->height);
        cloud_out->width = num_of_keypoints;
        cloud_out_rgb->width = num_of_keypoints;
        cloud_out->points.resize (cloud_out->width * cloud_out->height);
        cloud_out_rgb->points.resize (cloud_out->width * cloud_out->height);

        int i_x = get<0>(coordinate);
        int i_y = get<1>(coordinate);
        int j_x = get<2>(coordinate);
        int j_y = get<3>(coordinate);

        cloud_in->points[index].x = XYZcamera_i[0].at<float>(i_x, i_y);
        cloud_in_rgb->points[index].x = cloud_in->points[index].x;
        cloud_in->points[index].y = XYZcamera_i[1].at<float>(i_x, i_y);
        cloud_in_rgb->points[index].y = cloud_in->points[index].y;
        cloud_in->points[index].z = XYZcamera_i[2].at<float>(i_x, i_y);
        cloud_in_rgb->points[index].z = cloud_in->points[index].z;
        cloud_in_rgb->points[index].rgb = *reinterpret_cast<float*>(&rgb1);

        P->P3D_i.push_back(vector<float>{XYZcamera_i[0].at<float>(i_x, i_y),
                                         XYZcamera_i[1].at<float>(i_x, i_y),
                                         XYZcamera_i[2].at<float>(i_x, i_y)});
//        cout << "Keypoints: " << i_x << " " << i_y
//             << " XYZ: " << XYZcamera_i[0].at<float>(i_x, i_y) << " "
//             << XYZcamera_i[1].at<float>(i_x, i_y) << " "
//             << XYZcamera_i[2].at<float>(i_x, i_y) << endl;
        cloud_out->points[index].x = XYZcamera_j[0].at<float>(j_x, j_y);
        cloud_out_rgb->points[index].x = cloud_out->points[index].x;
        cloud_out->points[index].y = XYZcamera_j[1].at<float>(j_x, j_y);
        cloud_out_rgb->points[index].y = cloud_out->points[index].y;
        cloud_out->points[index].z = XYZcamera_j[2].at<float>(j_x, j_y);
        cloud_out_rgb->points[index].z = cloud_out->points[index].z;
        cloud_out_rgb->points[index].rgb = *reinterpret_cast<float*>(&rgb2);

        P->P3D_j.push_back(vector<float>{XYZcamera_j[0].at<float>(j_x, j_y),
                                         XYZcamera_j[1].at<float>(j_x, j_y),
                                         XYZcamera_j[2].at<float>(j_x, j_y)});

        index++;
    }

    for(auto coordinate : valid_keypoints){
        int i_x = get<0>(coordinate);
        int i_y = get<1>(coordinate);
        int j_x = get<2>(coordinate);
        int j_y = get<3>(coordinate);
        P->valid_i.push_back(Point(i_y, i_x)); // x cols
        P->valid_j.push_back(Point(j_y, j_x)); // y rows
    }

    image_i.release();
    image_j.release();
    depth_i.release();
    depth_j.release();
    descriptors_i.release();
    descriptors_j.release();

    cout << "number of valid points: " << P->valid_i.size() << endl;

    if(P->valid_i.size() >= 6)
    {
        // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_icp_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp (new pcl::PointCloud<pcl::PointXYZ>);
        *cloud_icp = *cloud_in;
        *cloud_icp_rgb = *cloud_in_rgb;

        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(cloud_in);
        icp.setInputTarget(cloud_out);
        icp.setMaxCorrespondenceDistance(0.05);
        icp.setTransformationEpsilon(1e-7);
        icp.setEuclideanFitnessEpsilon(0.001);
        icp.setMaximumIterations(1000);
        pcl::PointCloud<pcl::PointXYZ> Final;
        icp.align(Final);
        cout << "has converged:" << icp.hasConverged() << " score: " <<
                  icp.getFitnessScore() << std::endl;
        cout << icp.getFinalTransformation() << std::endl;

//        // /** Notice: type of getFinalTransformation() is float */
//        Eigen::Matrix4d transformation_matrix = icp.getFinalTransformation().cast<double>();
//        pcl::transformPointCloud (*cloud_in, *cloud_icp, transformation_matrix);
//
//        /** visualization */
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr test(new pcl::PointCloud<pcl::PointXYZRGB>);
////
//        /** convert xyz to xyzrgb */
//         for(int i = 0; i < cloud_icp->points.size(); i++){
//             cloud_icp_rgb->points[i].x = cloud_icp->points[i].x;
//             cloud_icp_rgb->points[i].y = cloud_icp->points[i].y;
//             cloud_icp_rgb->points[i].z = cloud_icp->points[i].z;
//             cloud_icp_rgb->points[i].rgb = *reinterpret_cast<float*>(&rgb3);
//         }
//
//         *test = *cloud_icp_rgb + *cloud_out_rgb;
//         // *test += *cloud_in_rgb;
//
//        // rgbVis(test);
//
////
//        /** icp_normal*/
//       pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_src (new pcl::PointCloud<pcl::PointNormal>);
//       pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_tgt (new pcl::PointCloud<pcl::PointNormal>);
//
//       pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> norm_est;
//       pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
//       norm_est.setSearchMethod (tree);
//       norm_est.setKSearch (30);
//
//       norm_est.setInputCloud (cloud_in);
//       norm_est.compute (*points_with_normals_src);
//       pcl::copyPointCloud (*cloud_in, *points_with_normals_src);
//
//       norm_est.setInputCloud (cloud_out);
//       norm_est.compute (*points_with_normals_tgt);
//       pcl::copyPointCloud (*cloud_out, *points_with_normals_tgt);
//
//       pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> reg;
//
//       // Set the point representation
//       // reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));
//
//       reg.setInputSource (points_with_normals_src);
//       reg.setInputTarget (points_with_normals_tgt);
//
//
//       pcl::PointCloud<pcl::PointNormal>::Ptr reg_result = points_with_normals_src;
//       reg.setMaximumIterations (500);
//       reg.setTransformationEpsilon (1e-5);
//       reg.setMaxCorrespondenceDistance (0.5);
//       reg.setEuclideanFitnessEpsilon(0.001);
//       reg.align (*reg_result);
//
//       Eigen::Matrix4f targetToSource;
////        // Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
//       targetToSource = reg.getFinalTransformation();
//       pcl::transformPointCloud (*cloud_in, *cloud_icp, transformation_matrix);
//
//       /** convert xyz to xyzrgb */
//       for(int i = 0; i < cloud_icp->points.size(); i++){
//           cloud_icp_rgb->points[i].x = cloud_icp->points[i].x;
//           cloud_icp_rgb->points[i].y = cloud_icp->points[i].y;
//           cloud_icp_rgb->points[i].z = cloud_icp->points[i].z;
//           cloud_icp_rgb->points[i].rgb = *reinterpret_cast<float*>(&rgb3);
//       }
//
//       *test = *cloud_icp_rgb + *cloud_out_rgb;
//       // *test += *cloud_in_rgb;
//
//       std::cout << "has converged:" << reg.hasConverged() << " score: " <<
//                 reg.getFitnessScore() << std::endl;
//       std::cout << reg.getFinalTransformation() << std::endl;
//
//       // rgbVis(test);
//
//        /** 2D matching */
//        Mat essential_matrix;
//        essential_matrix = findEssentialMat(P->valid_i, P->valid_j, K, RANSAC, 0.9, 2.0, noArray() );
//        // cout << "essential_matrix " << essential_matrix << endl;
//        Mat R, t;
//        recoverPose( essential_matrix, P->valid_i, P->valid_j, K, R, t, noArray());
//
////        P->R = R;
////        P->t = t;
//        cout << R << endl;
//        cout << t << endl;
//
//        Eigen::MatrixXd R_new;
//        Eigen::MatrixXd t_new;
//        cv2eigen(R, R_new);
//        cv2eigen(t, t_new);
//
//
//        Eigen::Matrix<double, 3, 4> trans1;
//        trans1 << R_new, t_new;
//        Eigen::Matrix4d trans;
//        trans << trans1, 0, 0, 0, 1;
//        cout << "trans: " << trans << endl;
//        P->R = R_new;
//        P->t = t_new;

//        /** NDT */
//        pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
//        ndt.setTransformationEpsilon (0.01);
//        // Setting maximum step size for More-Thuente line search.
//        ndt.setStepSize (0.1);
//        // Setting Resolution of NDT grid structure (VoxelGridCovariance).
//        ndt.setResolution (1.0);
//        // Setting max number of registration iterations.
//        ndt.setMaximumIterations (50);
//
//        pcl::VoxelGrid<pcl::PointXYZ> sor;
//        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_in (new pcl::PointCloud<pcl::PointXYZ>);
//        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_out (new pcl::PointCloud<pcl::PointXYZ>);
//
//        sor.setInputCloud (cloud_in_dense);
//        sor.setLeafSize (0.05f, 0.05f, 0.05f);
//        sor.filter (*cloud_filtered_in);
//
//        sor.setInputCloud (cloud_out_dense);
//        sor.filter (*cloud_filtered_out);
//
//        // Setting point cloud to be aligned.
//        ndt.setInputSource (cloud_filtered_in);
//        // Setting point cloud to be aligned to.
//        ndt.setInputTarget (cloud_filtered_out);
//        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>);
//
////        Eigen::AngleAxisf init_rotation (0.6931, Eigen::Vector3f::UnitZ ());
////        Eigen::Translation3f init_translation (1.79387, 0.720047, 0);
////        Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();
//// , icp.getFinalTransformation()
//        ndt.align(*output_cloud);
//        cout << "ndt has converged: " << ndt.hasConverged() << endl;
//        // cout << "score: " << ndt.getFitnessScore() << endl;
//        cout << "ndt: " << ndt.getFinalTransformation() << endl;

        /**Using Eigen, return from icp*/
//         P->R = transformation_matrix.topLeftCorner(3, 3);
//         P->t = transformation_matrix.topRightCorner(3, 1);
        // Eigen::Matrix4d transformation_matrix2;
        // -R_.transpose() * t_
        P->R = icp.getFinalTransformation().cast<double>().topLeftCorner(3, 3).transpose();
        P->t = -icp.getFinalTransformation().cast<double>().topLeftCorner(3, 3).transpose() * icp.getFinalTransformation().cast<double>().topRightCorner(3, 1);

       //
//        Eigen::Matrix4d transformation_matrix2;
//        transformation_matrix2 << 1.0, 0.0031, 0.0078, -0.0207,
//                                    -0.0031, 1.0, 0.0001, 0.0017,
//                                    -0.0078, -0.00001, 1.0, 0.0020,
//                                    0.0, 0.0, 0.0, 1.0;
//        pcl::transformPointCloud (*cloud_in, *cloud_icp, transformation_matrix2);
//        /** convert xyz to xyzrgb */
//        for(int i = 0; i < cloud_icp->points.size(); i++){
//            cloud_icp_rgb->points[i].x = cloud_icp->points[i].x;
//            cloud_icp_rgb->points[i].y = cloud_icp->points[i].y;
//            cloud_icp_rgb->points[i].z = cloud_icp->points[i].z;
//            cloud_icp_rgb->points[i].rgb = *reinterpret_cast<float*>(&rgb3);
//        }
//        *test = *cloud_icp_rgb + *cloud_out_rgb;
//        // *test += *cloud_in_rgb;
//
//        rgbVis(test);
//        while (!viewer->wasStopped ())
//        {
//            viewer->spinOnce (100);
//            std::this_thread::sleep_for(100ms);
//        }

//        essential_matrix.release();
//        R.release();
//        t.release();
        return P;
    }
    else
        cout << "Current num of valid points is less than 6." << endl;
    Eigen::MatrixXd tempCameraRtC2W(3, 4);
    tempCameraRtC2W <<  1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0;

    P->R = tempCameraRtC2W.topLeftCorner(3, 3);
    P->t = tempCameraRtC2W.topRightCorner(3, 1);

    return P;
}

void outputPly(const vector<Eigen::MatrixXd>& CameraRtC2W, const Mat& K,  pcl::PointCloud<pcl::PointXYZRGB>::Ptr& data_chunk_rgb){
    cout << "OutputPly..." << endl;
    int frameInterval = 10;
    int pointCount = round(
                          min( 640*480*0.5, 1000000.0 / (imagei2s.size()/frameInterval) )
                          );
    
    for(int frameID = 0; frameID < imagei2s.size(); frameID += frameInterval){
	    cout << "depth i: " << IDimage2depth[frameID] << " image i: " << ID2image[frameID] << endl;

    	FileStorage fs_i(IDimage2depth[frameID], FileStorage::READ);
    	// FileStorage fs_j(IDimage2depth[index_j], FileStorage::READ);
	
   	/** read depth, and the depth images are computed by MATLAB */
    	Mat depth(480, 640, CV_32F);
        fs_i["depth"] >> depth;
    	fs_i.release();
        vector<Mat> XYZcamera = depth2XYZcamera(depth, K);

        vector<Eigen::MatrixXd> XYZcamera_eigen(4);
        Eigen::MatrixXd XYZcamera_eigen_full(480*640, 3);
        for(int i = 0; i < 4; i++){
            int row = XYZcamera[i].rows;
	        int col = XYZcamera[i].cols;
    	    Eigen::MatrixXd temp_eigen(row, col);
    	    cv2eigen(XYZcamera[i], temp_eigen);
            XYZcamera_eigen[i] = temp_eigen;
            XYZcamera_eigen[i].resize(480*640, 1);
        }

        XYZcamera_eigen_full << XYZcamera_eigen[0],
                                XYZcamera_eigen[1],
                                XYZcamera_eigen[2];

        // XYZcamera_eigen_full.resize(480*640, 3);

        // cout << XYZcamera_eigen_full << endl;
        Eigen::MatrixXd XYZcamera_eigen_fullT;
        XYZcamera_eigen_fullT = XYZcamera_eigen_full.transpose();

        // cout << XYZcamera_eigen_fullT.middleCols(3658, 3674) << endl;

        vector<int> isValid;
        for(int i = 0; i < XYZcamera_eigen[3].size(); i++){
            if(XYZcamera_eigen[3](i) > 0)
                isValid.emplace_back(i);
        }

        cout << "Valid Size:" << isValid.size() << endl;
        // vector<int> isValid_rgb; 
        if(isValid.size() > pointCount){
            random_shuffle(isValid.begin(), isValid.end());
            isValid = vector<int>(isValid.begin(), isValid.begin() + pointCount);
        }
        /*
        if nargin<3 || ~isInverse
            Y3D = Rt(:,1:3) * X3D + repmat(Rt(:,4),1,size(X3D,2));
        else
            Y3D = Rt(:,1:3)' * (X3D - repmat(Rt(:,4),1,size(X3D,2)));
        end
        */
        Eigen::MatrixXd result;

        // cout << XYZcamera_eigen_fullT.rows() << " " << XYZcamera_eigen_fullT.cols() << endl;

        result = (CameraRtC2W[frameID].topLeftCorner(3, 3) * XYZcamera_eigen_fullT +
                 CameraRtC2W[frameID].topRightCorner(3, 1).replicate(1, XYZcamera_eigen_fullT.cols()) ).transpose();  // 3×3×3×（480×640） + 3×(480×640)

        Mat image = imread(ID2image[frameID]);
        // cout << image.type() << endl;
        
        unsigned int num = isValid.size();
        auto pre_width = (frameID == 0) ? 0 : data_chunk_rgb->width;
        data_chunk_rgb->width = data_chunk_rgb->width + num;
        cout << "Now Points: " << data_chunk_rgb->width << endl;
        data_chunk_rgb->points.resize (data_chunk_rgb->width * data_chunk_rgb->height);

        for(int index = 0; index < isValid.size(); index++){
            // cout << isValid[index] << endl;
            data_chunk_rgb->points[pre_width + index].x = result(isValid[index], 0);
            data_chunk_rgb->points[pre_width + index].y = result(isValid[index], 1);
            data_chunk_rgb->points[pre_width + index].z = result(isValid[index], 2);

            int i = isValid[index] % image.rows;
            int j = isValid[index] / image.rows;
            uint8_t r1(image.at<Vec3b>(i, j)[2]),
                    g1(image.at<Vec3b>(i, j)[1]), 
                    b1(image.at<Vec3b>(i, j)[0]);
            uint32_t rgb_temp = (static_cast<uint32_t>(r1) << 16 |
                    static_cast<uint32_t>(g1) << 8 | static_cast<uint32_t>(b1));
            data_chunk_rgb->points[pre_width + index].rgb = *reinterpret_cast<float*>(&rgb_temp);
        }
    }
}

void pose_estimation_3d3d( const vector<Point3f>& pts1, const vector<Point3f>& pts2)
{
    Point3f p1, p2; // center of mass
    int N = pts1.size();
    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];
        p2 += pts2[i]; 
    }
    p1 /= N; p2 /= N; // 取点中心
    vector<Point3f> q1(N), q2(N); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3d( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3d( q2[i].x, q2[i].y, q2[i].z )
        .transpose();
    }
    cout<<"W="<<W<<endl;
    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout<<"U="<<U<<endl;
    cout<<"V="<<V<<endl;

    Eigen::Matrix3d R_ = U*(V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d( p1.x, p1.y, p1.z ) - R_ * Eigen::Vector3d( p2.x, p2.y, p2.z );
    cout << "R_=: " << R_ << endl;
    cout << "t_=: " << t_ << endl;

    cout << "R_inv = \n" << R_.transpose() << endl;
    cout << "t_inv = \n" << -R_.transpose() * t_ << endl;
}

int main(int argc, char **argv){
     /** Read K and depth file
      */
     Mat K(3, 3, CV_32F);
     string dir_k = "/home/ian/CLionProjects/cuda/hotel_umd/maryland_hotel3/intrinsics.txt";
     K = readK(dir_k);

     string dir_depth = "/home/ian/CLionProjects/cuda/hotel_umd/maryland_hotel3/my_depth";
     create_depth_map(dir_depth);


     string dir = "/home/ian/CLionProjects/cuda/hotel_umd/maryland_hotel3/image_big";

     clock_t sTime = clock();
     cout << "-----------------------------------------" << endl;
     cout << "Sift Begin..." << endl;
     DetectComputeimage(dir);
     cout << "Sift End in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " seconds." << endl;
     cout << endl;

     sTime = clock();
     cout << "-----------------------------------------" << endl;
     cout << "BOW Training Begin..." << endl;
     Mat CodeBook;
     CodeBook = create_codebook();
     auto number_of_words = CodeBook.rows;
     cout << "BOW Training End " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " seconds." <<  endl;

     cout << "-----------------------------------------" << endl;
     cout << "Get histogram(training_data and train_label)..." << endl;
 //    for(int i = 0; i < category.size(); i++){
         // cout << dir+ "/" +i << endl;
         create_bow_histogram_features(CodeBook, dir);
 //    }
     cout << "Get histogram end " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " seconds." << endl;

 //    cout << "-----------------------------------------" << endl;
 //    cout << "Normalize histogram..." << endl;
 //    histograms = normalize_histograms(histograms);
 //    cout << "Normalize historam ends." << endl;
 //
 //
 //    Mat scores;
 //    scores = histograms * histograms.t();
 //    FileStorage fs_scores("scores.yml", FileStorage::WRITE);
 //    FileStorage fs_histograms("histograms.yml", FileStorage::WRITE);
 //    cout << "histograms type: " << histograms.type() << endl;
 //    fs_histograms << "histograms" << histograms;
 //    cout << "scores type: " << scores.type() << endl;
 //    fs_scores << "scores" << scores;
 //    fs_scores.release();
 //    fs_histograms.release();
 //
 //
 ////    FileStorage fs3("scores.yml", FileStorage::READ);
 ////    Mat scores;
 ////    fs3["scores"] >> scores;
 ////    fs3.release();
 //
 //    Mat scores_dst, temp_dst;
 //    Mat temp = Mat::ones(scores.rows, scores.cols, CV_8UC1);
 //    for(int i = 0; i < scores.rows; i++)
 //        temp.at<u_char>(i, i) = 0;
 //    // cout << temp;
 //    distanceTransform(temp, temp_dst, DIST_L2, DIST_MASK_PRECISE);
 //
 //    temp.release();
 //////    for(int i = 0; i < scores.rows; i++)
 //////        for(int j = 0; j < scores.cols; j++ )
 //////            if(temp_dst.at<uchar>(i, j) > '1')
 //////                temp_dst.at<uchar>(i, j) = '1';
 ////     cout << temp_dst << endl;
 ////    cout << "temp_dest flag" << endl;
 //    for(int i = 0; i < scores.rows; i++)
 //        for(int j = 0; j < scores.cols; j++ )
 //        {
 //            if(j >= i)
 //                scores.at<float>(i, j) = 0.0;
 //            else{
 //                if(static_cast<float>(temp_dst.at<uchar>(i, j)) > 1.0)
 //                    scores.at<float>(i, j) = 1.0 / 30.0 * scores.at<float>(i, j);
 //                else
 //                    scores.at<float>(i, j) = static_cast<float>(temp_dst.at<uchar>(i, j)) / 30.0 * scores.at<float>(i, j);
 //            }
 //        }
 //
 //    cout << "scores ends. ";
 ////    cout << scores;
 ////    scores = scores_dst;
 //
 //    GaussianBlur( scores, scores_dst, Size( 5, 5 ), 0, 0 );
 //    FileStorage fs2("scores_dst.yml", FileStorage::WRITE);
 //    fs2 << "scores_dst" << scores_dst;
 //    fs2.release();
 //
 //    Mat t = nonmaxup(scores_dst, 7);
 //
 //    /**
 //     * note: this should also be sorted and delete the low value score matchs
 //     */
 //    vector<int> cameras_i;
 //    vector<int> cameras_j;
 //    for(int i = 0; i < t.rows; i++)
 //        for(int j = 0; j < t.cols; j++){
 //            if(t.at<float>(i, j) > 0.001) {  // threshold
 //                cameras_i.push_back(i);
 //                cameras_j.push_back(j);
 //            }
 //        }
 //    scores.release();
 //    scores_dst.release();
 //    temp_dst.release();
 //    t.release();
 //    histograms.release();
 //    allDescriptors.release();
     // PCL 点云库 ICP方法

 //    vector<pairs*> MatchPairsLoop;
 //    for(int i = 0; i < cameras_i.size(); i++)
 //    {
 //        // cout << i << " " << cameras_i.size() - 1 << endl;
 //        pairs* temp = align2view(cameras_i[i], cameras_j[i], K);
 //        if(temp != nullptr)
 //            MatchPairsLoop.push_back(temp);
 //    }

 //    align2view(9, 5, K);

     cout << "Time based MatchPairs Begin..." << endl;
     // int NullPtr = 0;
     vector<pairs*> MatchPairs;
     for(int i = 0; i < imagei2s.size() - 1; i++) {
         pairs *temp = align2viewTimebased(i, i + 1, K);
         // if(temp != nullptr) {
         MatchPairs.push_back(temp);
         // NullPtr = 1;
         //}
     }
     cout << "Time based MatchPairs end " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " seconds." << endl;


     Eigen::MatrixXd tempCameraRtC2W(3, 4);
     tempCameraRtC2W <<  1, 0, 0, 0,
                         0, 1, 0, 0,
                         0, 0, 1, 0;

     vector<Eigen::MatrixXd> CameraRtC2W( imagei2ds.size() );
     for(int i = 0; i < imagei2ds.size(); i++) {
         CameraRtC2W[i] = tempCameraRtC2W;
     }
   //  cout << "CameraRtC2W Begin..." << endl;
     for(int i = 0; i < imagei2s.size() - 1; i++) {
         /* cameraRtC2W(:,1:3,frameID) * MatchPairs{frameID}.Rt(:,1:3)
          * cameraRtC2W(:,1:3,frameID) * MatchPairs{frameID}.Rt(:,4) + cameraRtC2W(:,4,frameID)];*/
    //     cout << i << endl;
         // if(MatchPairs[i] == nullptr)
         //     continue;
 //        cout << MatchPairs[i]->R << endl;
 //        cout << MatchPairs[i]->t << endl;
         Eigen::Matrix3d A = CameraRtC2W[i].topLeftCorner(3, 3) * MatchPairs[i]->R;
         Eigen::MatrixXd B = CameraRtC2W[i].topLeftCorner(3, 3) * MatchPairs[i]->t + CameraRtC2W[i].topRightCorner(3, 1);
         CameraRtC2W[i + 1] << A, B;
 //        cout << i <<endl;
 //        cout << CameraRtC2W[i + 1] << endl;
     }
   //  cout << "CamerRtC2W end. " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " seconds." << endl;


     pcl::PointCloud<pcl::PointXYZRGB>::Ptr data_chunk_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
     data_chunk_rgb->height = 1;
     outputPly(CameraRtC2W, K, data_chunk_rgb);
     pcl::io::savePCDFileASCII ("data_chunk_rgb.pcd", *data_chunk_rgb);
     rgbVis(data_chunk_rgb);




    return 0;
}
