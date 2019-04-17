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