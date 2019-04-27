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