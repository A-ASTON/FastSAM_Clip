#pragma once
#include "fastsam_instance/common-fastsam.h"
#include "fastsam_instance/fastsam.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

namespace SamToClip {
class FastSamInstance {
public:
    FastSamInstance(const std::string& engine_file_path);
    ~FastSamInstance();

    void infer(const cv::Mat& img);
    void format_results(); // 根据内部objs_计算
private:
    // trt module related只保留一个FastSAM实例
    std::unique_ptr<FastSam> fastsam_;
    std::vector<Object> objs_;
    int cnt_;
    double time_sum_;
    cv::Mat color_mask_, image_;
    std::vector<annotation> format_results_;
};

}