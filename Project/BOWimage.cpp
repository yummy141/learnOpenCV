#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>

//#include <Windows.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <omp.h>
#include <sys/stat.h>
#include <dirent.h>
// #define NUM_THREADS 8

using namespace std;
using namespace cv;
using namespace cv::ml;

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

const int TESTING_PERCENT_PER = 7;
const int DICT_SIZE = 230;	//80 word per class

Mat allDescriptors;

void DetectComputeimage(const string& folder_path) {

    DIR *dp;
    struct dirent *dirp;
    struct stat filestat;
    string filepath;

    dp = opendir( folder_path.c_str() );
    vector<string> res;
    while (dirp = readdir( dp )){

        filepath = folder_path + "/" + dirp->d_name;
        // If the file is a directory (or is in some way invalid) we'll skip it
        if (stat( filepath.c_str(), &filestat ) || S_ISDIR( filestat.st_mode )){
                continue;
            }

        if (filepath.find(".sift") == -1){
            cout <<"processing: " << filepath << endl;

            Ptr<xfeatures2d::SIFT> siftptr;
            siftptr = xfeatures2d::SIFT::create();

            Mat img = imread(filepath);
            vector<KeyPoint> keypoints;
            Mat descriptors;
            siftptr->detectAndCompute(img, noArray(), keypoints, descriptors);

            allDescriptors.push_back(descriptors);
           //  allDescPerImg.emplace_back(descriptors);

        }
        // cout << filepath << endl;
    }
    return;


//    cout << keypoints.size() << endl;
//    cout << descriptors.size() << endl;
//
//    allDescPerImgNum++;
}

Mat create_codebook(){

    auto number_of_descriptors = static_cast<float>(allDescriptors.rows);
    auto number_of_words = static_cast<int>(sqrt(number_of_descriptors));
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


map<string,Mat> classes_training_data;
Mat train_data;
Mat train_label;
void create_bow_histogram_features(Mat codebook, string folder_path, const int label){
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

    vector<string> res;
    while (dirp = readdir( dp )){
        filepath = folder_path + "/" + dirp->d_name;
        // If the file is a directory (or is in some way invalid) we'll skip it
        if (stat( filepath.c_str(), &filestat ) || S_ISDIR( filestat.st_mode )){
            continue;
        }

        if (filepath.find(".sift") == -1 && filepath.find(".bmp")){
            cout <<"processing: " << filepath << endl;

            Mat img = imread(filepath);
           //  Mat grayimg;
           //  cvtColor(imread(img, grayimg, CV_BGR2GRAY);
            Mat response_hist;

            vector<KeyPoint> keypoints;
            detector->detect(img, keypoints);
            cout << "keypoints size: " << keypoints.size() << endl;
            bowide.compute(img, keypoints, response_hist);

            // classes_training_data[filepath] = response_hist;

            train_data.push_back(response_hist);
            train_label.push_back(Mat(1, 1, CV_32SC1, label));

        }
        // cout << filepath << endl;
    }
    return;
}


vector<string> category;
Ptr<SVM> svm = SVM::create();
void predict(Mat codebook, string folder_path){
    Ptr<FeatureDetector > detector = xfeatures2d::SIFT::create(); //detector
    Ptr<DescriptorExtractor > extractor = xfeatures2d::SIFT::create();//  extractor;
//    Ptr<DescriptorExtractor > extractor(
//            new OpponentColorDescriptorExtractor(
//                    Ptr<DescriptorExtractor>(new SurfDescriptorExtractor())
//            )
//    );
    Ptr<DescriptorMatcher> matcher(new BFMatcher());
    BOWImgDescriptorExtractor bowide(extractor, matcher);
    bowide.setVocabulary(codebook);

    DIR *dp;
    struct dirent *dirp;
    struct stat filestat;
    string filepath;
    dp = opendir( folder_path.c_str() );

    while (dirp = readdir( dp )){
        filepath = folder_path + "/" + dirp->d_name;
        // If the file is a directory (or is in some way invalid) we'll skip it
        if (stat( filepath.c_str(), &filestat ) || S_ISDIR( filestat.st_mode )){
            continue;
        }

        if (filepath.find(".sift") == -1 && filepath.find(".bmp") != -1){
            cout <<"processing: " << filepath << endl;

            Mat img = imread(filepath);
            Mat response_hist;

            vector<KeyPoint> keypoints;
            detector->detect(img, keypoints);
            cout << "keypoints size: " << keypoints.size() << endl;
            bowide.compute(img, keypoints, response_hist);
            int predict_label = static_cast<int>(svm->predict(response_hist));
            cout << "predicit label is " << predict_label << endl;
            cout << filepath << " is predicted as "<< category[predict_label] << endl;

        }
        // cout << filepath << endl;
    }
    return;
}


int main(int argc, char **argv){
//    clock_t sTime = clock();
//    cout << "Reading inputs..." << endl;

//    cout << "-> Reading, Detect and Describe input in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;


    string dir = "../dataset";

    category = get_categories(dir);
    auto ncat = category.size();
    if(ncat < 1){
        cerr << "Only " << ncat  << " categories found. Wrong path?" << endl;
    }
    cout << "found " << ncat << " folders / categories:" << endl;
    for(const auto& i : category)
        cout << i << endl;


    cout << "-----------------------------------------" << endl;
    cout << "Sift Begin..." << endl;
    for(const auto& i : category){
        // cout << dir+ "/" +i << endl;
        DetectComputeimage(dir+ "/" +i);
    }
    cout << "Sift End." << endl;
    cout << endl;


    cout << "-----------------------------------------" << endl;
    cout << "BOW Training Begin..." << endl;
    Mat CodeBook;
    CodeBook = create_codebook();
    auto number_of_words = CodeBook.rows;
    cout << "BOW Training End." << endl;


    cout << "-----------------------------------------" << endl;
    cout << "Get histogram(training_data and train_label)..." << endl;
    for(int i = 0; i < category.size(); i++){
        // cout << dir+ "/" +i << endl;
        create_bow_histogram_features(CodeBook, dir + "/" + category[i], i);
    }
    cout << "Get histogram end." << endl;

    cout << "-----------------------------------------" << endl;
    cout << "SVM_training..." << endl;
    svm->setKernel(SVM::RBF);
    // svm->setKernel(SVM::LINEAR);
    // svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e4, 1e-6));
    svm->setGamma(0.01);
    svm->setC(100.0);

    Ptr<TrainData> td = TrainData::create(train_data, ROW_SAMPLE, train_label);
    svm->train(td);
    FileStorage fs2("train_data.yml", FileStorage::WRITE);
    fs2 << "train_data" << train_data;
    fs2.release();

    FileStorage fs3("train_label.yml", FileStorage::WRITE);
    fs3 << "train_label" << train_label;
    fs3.release();
    cout << "SVM_training end." << endl;


    cout << "-----------------------------------------" << endl;
    cout << "SVM_predicting..." << endl;
    predict(CodeBook, dir);
    cout << "SVM_predicting done" << endl;

    return 0;
}
