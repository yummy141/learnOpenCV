
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
//#include <Windows.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <omp.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstdio>
// #define NUM_THREADS 8

using namespace std;
using namespace cv;
using namespace cv::ml;
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

//            drawKeypoints( img, keypoints, img_keypoints );
//            imshow(a, img);
            cout << "descriptors size: " << descriptors.size << endl;
            allDescriptors.push_back(descriptors);
           //  allDescPerImg.emplace_back(descriptors);

        }
        // cout << filepath << endl;
    }
    return;

//    allDescPerImgNum++;
}

Mat create_codebook(){

    auto number_of_descriptors = static_cast<float>(allDescriptors.rows);
    int number_of_words = 1000;
    // auto number_of_words = static_cast<int>(sqrt(number_of_descriptors));
    cout << "Total descriptors: " << number_of_descriptors << endl;

    FileStorage fs("training_descriptors.yml", FileStorage::WRITE);
    fs << "training_descriptors" << allDescriptors;
    fs.release();

    cout << "Now descriptors: " << number_of_words << endl;
    BOWKMeansTrainer bowtrainer(number_of_words); //num clusters
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
    Mat R;
    Mat t;
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
             */
            imagei2s[file_num] = filepath;

            int FrameID, _Timestamp;
            sscanf(dirp->d_name, "%d-%d.png", &FrameID, &_Timestamp);
            ID2image[FrameID] = filepath;
            file_num++;

            imagei2ds[file_num] = IDimage2depth[FrameID];


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

Mat normalize_histograms(Mat h){
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


pairs* align2view(int i, int j, const Mat& K){
    /**
     * i,j : filenumber
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
    pairs test;
    test.i = i;
    test.j = j;

    pairs *P = nullptr;
    P->i = i;
    P->j = j;

    Mat image_i = imread(imagei2s[i]);
    Mat image_j = imread(imagei2s[j]);

    FileStorage fs_i(IDimage2depth[i], FileStorage::READ);
    FileStorage fs_j(IDimage2depth[j], FileStorage::READ);

    Mat depth_i;
    Mat depth_j;
    fs_i["depth"] >> depth_i;
    fs_j["depth"] >> depth_j;

    fs_i.release();
    fs_j.release();

//    //FileStorage depthW("depth.yml", FileStorage::WRITE);
//    // depthW << "depth" << depth_i;
//    // depthW.release();
//    // Mat depth_i = depthRead(test_A);
//    cout << "depth_i okay." << endl;
////    Mat depth_i_new(depth_i.rows, depth_i.cols, CV_32F);
////    depth_i.convertTo(depth_i_new, CV_32F);
////    depth_i_new /= 1000.0;
    // depth_i_new.release();
    // cout << "j: " << j << endl;
    // cout << IDimage2depth[j] << endl;
//    Mat depth_j = depthRead(IDimage2depth[j]);
//    // Mat depth_j = depthRead(test_B);
//    Mat depth_j_new(depth_j.rows, depth_j.cols, CV_32F);
//    depth_j.convertTo(depth_j_new, CV_32F);
//    depth_j_new /= 1000.0;
//    cout << "depth_j ends.";
//    cout << "depth okay." << endl;
//    FileStorage depthJ("depth_j.yml", FileStorage::WRITE);
//    depthJ << "depth_j" << depth_j_new;
//    depthJ.release();
//
//    depth_j.release();
//    depth_j_new.release();
    /** compute xyz CAMERA from depth
     */
    Mat X,Y;
    meshgrid(Range(1, 640), Range(1, 480), X, Y); // 480 * 640

// depth2XYZcamera(K, depth)
//    Mat test1 = (X - K.at<float>(1,3));
//    cout << test1.type() << endl;
//    cout << depth_i_new.type() << endl;
//    FileStorage test1File("test1.yml", FileStorage::WRITE);
//    FileStorage depthINewFile("depthINewFile.yml", FileStorage::WRITE);
//    depthINewFile << "depth_i_new" << depth_i_new;
//    test1File << "test1" << test1;
//    depthINewFile.release();
//    test1File.release();
//    Mat test = test1.mul(depth_i_new);
    Mat XYZcamera_i1 = (X - K.at<float>(1,3)).mul(depth_i) / K.at<float>(1,1);
    Mat XYZcamera_i2 = (Y - K.at<float>(2,3)).mul(depth_i) / K.at<float>(2,2);
    Mat XYZcamera_i3 = depth_i;
//    Mat XYZcamera_i4(XYZcamera_i3.rows, XYZcamera_i4.cols, CV_8UC1);
//    for(int i = 0; i < XYZcamera_i3.rows; i++)
//        for(int j = 0; j < XYZcamera_i3.cols; j++){
//            if(depth_i_new.at<float>(i, j) == 0)
//                XYZcamera_i4.at<uchar>(i, j) = 1;
//            else
//                XYZcamera_i4.at<uchar>(i, j) = 0;
//        }

    Mat XYZcamera_j1 = (X - K.at<float>(1,3)).mul(depth_j) / K.at<float>(1,1);
    Mat XYZcamera_j2 = (Y - K.at<float>(2,3)).mul(depth_j) / K.at<float>(2,2);
    Mat XYZcamera_j3 = depth_j;
//    Mat XYZcamera_j4(XYZcamera_j3.rows, XYZcamera_j4.cols, CV_8UC1);
//    for(int i = 0; i < XYZcamera_j3.rows; i++)
//        for(int j = 0; j < XYZcamera_j3.cols; j++){
//            if(depth_j_new.at<float>(i, j) == 0)
//                XYZcamera_j4.at<uchar>(i, j) = 1;
//            else
//                XYZcamera_j4.at<uchar>(i, j) = 0;
//        }
   //  cout << XYZcamera_j4 << endl;

//
    /** detect sift keypoints location and descriptors
     */
    // cout << image_i << endl;
    Ptr<xfeatures2d::SIFT>siftptr = xfeatures2d::SIFT::create();
    vector<KeyPoint> keypoints_i, keypoints_j;
    Mat descriptors_i, descriptors_j;
    siftptr->detectAndCompute(image_i, noArray(), keypoints_i, descriptors_i);
    siftptr->detectAndCompute(image_j, noArray(), keypoints_j, descriptors_j);

    /** sift matching
     */
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors_i, descriptors_j, knn_matches, 2 );

    float dMyRatihoThresh =  0.6f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
        if (knn_matches[i][0].distance < dMyRatihoThresh * knn_matches[i][1].distance)
            good_matches.push_back(knn_matches[i][0]);

    /* plot Matches
    //    Mat img_matches;
    //    drawMatches( image_i, keypoints_i, image_j, keypoints_j, good_matches, img_matches, Scalar::all(-1),
    //                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //    const string testS2 = "Test";
    //    imshow(testS2, img_matches);
    //    waitKey(0);
    */
    cout << "Question is under this line 540" << endl;
    /** find keypoints_match
    */
    vector<KeyPoint> keypoints_match_i, keypoints_match_j;
    for(auto match : good_matches){
        // cout << match.queryIdx << " " << match.trainIdx << " "<< endl;
        keypoints_match_i.push_back(keypoints_i[match.queryIdx]);
        keypoints_match_j.push_back(keypoints_j[match.trainIdx]);
    }

    set<tuple<int, int, int, int>> valid_keypoints;
    for(int i = 0; i < keypoints_i.size(); i++){
        int temp_i_x = round(keypoints_match_i[i].pt.x);
        int temp_i_y = round(keypoints_match_i[i].pt.y);
        int temp_j_x = round(keypoints_match_j[i].pt.x);
        int temp_j_y = round(keypoints_match_j[i].pt.y);

        if(temp_i_x >= 0 && temp_i_x < image_i.rows && temp_i_y >=0 && temp_i_y < image_i.cols
           && temp_j_x >= 0 && temp_j_y < image_j.rows && temp_j_y >= 0 && temp_j_y < image_j.cols
           && depth_i.at<float>(temp_i_x, temp_i_y) != 0
           && depth_j.at<float>(temp_j_x, temp_j_y) != 0)
        {
            valid_keypoints.insert(make_tuple(temp_i_x, temp_i_y, temp_j_x, temp_j_y));
            // valid_j_keypoints.insert({temp_j_x, temp_j_y});

            // cout << temp_i_x << " " << temp_i_y << endl;
        }
    }

    /** use matched kepoints to compute R and t
    */
    // vector<Point> valid_i;
    // vector<Point> valid_j;

    for(auto coordinate : valid_keypoints){
        int i_x = get<0>(coordinate);
        int i_y = get<1>(coordinate);
        int j_x = get<2>(coordinate);
        int j_y = get<3>(coordinate);
        P->valid_i.push_back(Point(i_x, i_y));
        P->valid_j.push_back(Point(j_x, j_y));
    }

    cout << "number of valid points: " << P->valid_i.size() << endl;
    if(P->valid_i.size() >= 6)
    {
        Mat essential_matrix;
        essential_matrix = findEssentialMat(P->valid_i, P->valid_j, K, RANSAC, 0.999, 1.0, noArray() );
        // cout << "essential_matrix " << essential_matrix << endl;
        Mat R, t;
        recoverPose( essential_matrix, P->valid_i, P->valid_j, K, R, t, noArray());
        P->R = R;
        P->t = t;
        return P;
    }
    else
        cout << "Current num of valid points is less than 6." << endl;


    return nullptr;
    // cout << "R " << R << endl;
    // cout << "t " << t << endl;

//    vector<vector<float>> P3D_i;
//    vector<vector<float>> P3D_j;
//    for(auto coordinate : valid_keypoints){
//        int i_x = get<0>(coordinate);
//        int i_y = get<1>(coordinate);
//        int j_x = get<2>(coordinate);
//        int j_y = get<3>(coordinate);
//        P3D_i.push_back(vector<float>{XYZcamera_i1.at<float>(i_x, i_y),
//                                      XYZcamera_i2.at<float>(i_x, i_y),
//                                      XYZcamera_i3.at<float>(i_x, i_y)});
//        P3D_j.push_back(vector<float>{XYZcamera_j1.at<float>(j_x, j_y),
//                                      XYZcamera_j2.at<float>(j_x, j_y),
//                                      XYZcamera_j3.at<float>(j_x, j_y)});
//    }

    /** align RANSAC
     */
}



int main(int argc, char **argv){
    /** Read K and depth file
     */
    Mat K(3, 3, CV_32F);
    string dir_k = "/home/ian/CLionProjects/cuda/hotel_umd/maryland_hotel3/intrinsics.txt";
    K = readK(dir_k);



    string dir_depth = "/home/ian/CLionProjects/cuda/hotel_umd/maryland_hotel3/my_depth";
    create_depth_map(dir_depth);

//
//    align2view(1, 11, K);

    string dir = "/home/ian/CLionProjects/cuda/hotel_umd/maryland_hotel3/image";

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

    cout << "-----------------------------------------" << endl;
    cout << "Normalize histogram..." << endl;
    histograms = normalize_histograms(histograms);
    cout << "Normalize historam ends." << endl;


    Mat scores;
    scores = histograms * histograms.t();
    FileStorage fs_scores("scores.yml", FileStorage::WRITE);
    FileStorage fs_histograms("histograms.yml", FileStorage::WRITE);
    cout << "histograms type: " << histograms.type() << endl;
    fs_histograms << "histograms" << histograms;
    cout << "scores type: " << scores.type() << endl;
    fs_scores << "scores" << scores;
    fs_scores.release();
    fs_histograms.release();


//    FileStorage fs3("scores.yml", FileStorage::READ);
//    Mat scores;
//    fs3["scores"] >> scores;
//    fs3.release();

    Mat scores_dst, temp_dst;
    Mat temp = Mat::ones(scores.rows, scores.cols, CV_8UC1);
    for(int i = 0; i < scores.rows; i++)
        temp.at<u_char>(i, i) = 0;
    // cout << temp;
    distanceTransform(temp, temp_dst, DIST_L2, DIST_MASK_PRECISE);

    temp.release();
////    for(int i = 0; i < scores.rows; i++)
////        for(int j = 0; j < scores.cols; j++ )
////            if(temp_dst.at<uchar>(i, j) > '1')
////                temp_dst.at<uchar>(i, j) = '1';
    cout << temp_dst << endl;
    cout << "temp_dest flag" << endl;
    for(int i = 0; i < scores.rows; i++)
        for(int j = 0; j < scores.cols; j++ )
        {
            if(j >= i)
                scores.at<float>(i, j) = 0.0;
            else{
                if(static_cast<float>(temp_dst.at<uchar>(i, j)) > 1.0)
                    scores.at<float>(i, j) = 1.0 / 30.0 * scores.at<float>(i, j);
                else
                    scores.at<float>(i, j) = static_cast<float>(temp_dst.at<uchar>(i, j)) / 30.0 * scores.at<float>(i, j);
            }
        }

    cout << "scores ends. ";
//    cout << scores;
//    scores = scores_dst;

    GaussianBlur( scores, scores_dst, Size( 5, 5 ), 0, 0 );
    FileStorage fs2("scores_dst.yml", FileStorage::WRITE);
    fs2 << "scores_dst" << scores_dst;
    fs2.release();

    Mat t = nonmaxup(scores_dst, 7);

    /**
     * note: this should also be sorted and delete the low value score matchs
     */
    vector<int> cameras_i;
    vector<int> cameras_j;
    for(int i = 0; i < t.rows; i++)
        for(int j = 0; j < t.cols; j++){
            if(t.at<float>(i, j) > 0.001)  // threshold
                cameras_i.push_back(i);
                cameras_j.push_back(j);
        }
//    scores.release();
//    scores_dst.release();
//    temp_dst.release();
//    t.release();
//    histograms.release();
//    allDescriptors.release();
    // PCL 点云库 ICP方法

    vector<pairs*> MatchPairsLoop;
    for(int i = 0; i < cameras_i.size(); i++)
    {
        cout << i << " " << cameras_i.size() - 1 << endl;
        pairs* temp = align2view(cameras_i[i], cameras_j[i], K);
        if(temp != nullptr)
            MatchPairsLoop.push_back(temp);
    }

//     align2view(9, 0, K);
//     align2view(cameras_i[0], cameras_j[0], K);


//    Mat test = imread("/home/ian/CLionProjects/cuda/hotel_umd/maryland_hotel3/depth/0000001-000000000000.png", CV_16U);
//    cout << test << endl;
//    cout << "-----------------------------------------" << endl;
//    cout << "SVM_training..." << endl;
//    svm->setKernel(SVM::RBF);
//    // svm->setKernel(SVM::LINEAR);
//    // svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e4, 1e-6));
//    svm->setGamma(0.01);
//    svm->setC(100.0);
//
//    Ptr<TrainData> td = TrainData::create(train_data, ROW_SAMPLE, train_label);
//    svm->train(td);
//    FileStorage fs2("train_data.yml", FileStorage::WRITE);
//    fs2 << "train_data" << train_data;
//    fs2.release();
//
//    FileStorage fs3("train_label.yml", FileStorage::WRITE);
//    fs3 << "train_label" << train_label;
//    fs3.release();
//    cout << "SVM_training end." << endl;
//
//
//    cout << "-----------------------------------------" << endl;
//    cout << "SVM_predicting..." << endl;
//    predict(CodeBook, dir);
//    cout << "SVM_predicting done" << endl;

    return 0;
}
