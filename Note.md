## 安装各种库的网站指南
> [kezunlin](https://kezunlin.me/)

## OpenCV和vl_sift的区别
- vl_sift and vl_dsift
    - > [reference](https://stackoverflow.com/questions/41038881/sift-descriptors-values-opencv-vs-vlfeat)
    - vl_sift能得到int类型的descriptor
    - vl_dsift能得到float类型的descriptor

## 如何访问一个Mat中的元素 
```c++
float *t1 = (float*)depth_i.data;
int _stride = depth_i.step;
float t2 = t1[temp_i_x * _stride + temp_i_y];

//或者
float t1 = depth_i.at<float>(temp_i_x, temp_i_y);
```


## Eigen拼接矩阵
> [stackoverflow](https://stackoverflow.com/questions/21496157/eigen-how-to-concatenate-matrix-along-a-specific-dimension)
```c++
You can use the comma initializer syntax for that.

Horizontally:

MatrixXd C(A.rows(), A.cols()+B.cols());
C << A, B;

Vertically:

// eigen uses provided dimensions in declaration to determine
// concatenation direction
MatrixXd D(A.rows()+B.rows(), A.cols()); // <-- D(A.rows() + B.rows(), ...)
D << A, B; // <-- syntax is the same for vertical and horizontal concatenation

For readability, one might format vertical concatenations with whitespace:

D << A,
     B; // <-- But this is for readability only. 


```

## Eigen矩阵读入
```c++
vector<pairs*> MatchPairs;
ifstream fin;
fin.open("matchpairs.in");
for (int n = 0; n < 74; n++){
pairs* tp = new pairs;
double t;
for (int i = 0; i < 3; i++){
    for (int j = 0; j < 4; j++){
        fin >> t;
        if(j < 3)
            tp->R(i, j) = t;
        else
            tp->t(i, 0) = t;
    }
}
cout << tp->R << endl;
cout << tp->t << endl;
MatchPairs.emplace_back(tp);
}
fin.close();
```

## icp测试
```c++
    ifstream fin3;
    fin3.open("ransac.in");
    vector<int> isValid;
    for(int i = 0; i < 153; i++){
        int t3;
        fin3 >> t3;
        isValid.emplace_back(t3);
    }
    fin3.close();

    // std::vector<Matrix4f> T0;//创建储存多个矩阵的vector对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

    cloud_in->height = 1; cloud_in->is_dense = false;
    cloud_out->height = 1; cloud_out->is_dense = false;
    cloud_in->width = 153; cloud_in->points.resize(153);
    cloud_out->width = 153; cloud_out->points.resize(153);
    ifstream fin1, fin2;
    fin1.open("P3D_i.in");
    fin2.open("P3D_j.in");
    
    for(int n = 0; n < 3; n++){
        int index = 0;
        for (int i = 1; i <= 165; i++){
            vector<int>::iterator it;
            it = find(isValid.begin(), isValid.end(), i);
            double t1, t2;
            fin1 >> t1;
            fin2 >> t2;
            if(it != isValid.end())
            {
                if(n == 0){
                    cloud_in->points[index].x = t1;
                    cloud_out->points[index].x = t2;
                }
                if(n == 1){
                    cloud_in->points[index].y = t1;
                    cloud_out->points[index].y = t2;
                }
                if(n == 2){
                    cloud_in->points[index].z = t1;
                    cloud_out->points[index].z = t2;
                }
                index++;
            }

        }
    }

    fin1.close();
    fin2.close();
    cout << cloud_in->points[0].x << " " << cloud_in->points[0].y << " " << cloud_in->points[0].z << endl;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud_in);
    icp.setInputTarget(cloud_out);
    icp.setMaxCorrespondenceDistance(0.05);
    icp.setTransformationEpsilon(1e-10);
    icp.setEuclideanFitnessEpsilon(1e-4);
    icp.setMaximumIterations(1000);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    cout << "has converged:" << icp.hasConverged() << " score: " <<
                icp.getFitnessScore() << std::endl;
    cout << icp.getFinalTransformation() << std::endl;
```