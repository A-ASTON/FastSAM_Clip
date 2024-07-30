#pragma once
#include <opencv2/opencv.hpp>

namespace SamToClip {
struct Segment{
    size_t id = -1; // id
    cv::Rect_<float> bbox; // 原图上的obj.rect bbox
    cv::Mat img; // 大小和原图一样的通过mask得到分割区域的图片 obj.boxMask
    float score = 0; // obj.prob
    float area = 0; // mask 的面积
    float *embedding = nullptr; // use clip to encode
    int embedding_dim = 0;
    bool valid = false;
};

inline void checkNorm(const float* embedding, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += pow(embedding[i], 2);
    }
    std::cout << "sum = " << sum << std::endl;
}


}
