#pragma once
#include <opencv2/opencv.hpp>

namespace SamToClip {
struct Segment{
    size_t id; // id
    cv::Rect_<float> bbox; // 原图上的obj.rect bbox
    cv::Mat img; // 大小和原图一样的通过mask得到分割区域的图片 obj.boxMask
    float score; // obj.prob
    float area; // mask 的面积
    float *embedding = nullptr; // use clip to encode
    int embedding_dim = 0;
};

}
