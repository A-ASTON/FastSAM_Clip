#pragma once
#include "fastsam_instance/common-fastsam.h"
#include "fastsam_instance/fastsam.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

namespace SamToClip {
class FastSamInstance {
public:
    FastSamInstance(const std::string& engine_file_path);
    ~FastSamInstance();

    void infer(const cv::Mat& img);
    void crop_images(bool filter = false); //根据后处理得到的objs_，更新segments_
    Segment segment_image(const Object& obj, const int id);
    std::vector<Segment> get_segments();
private:
    // trt module related只保留一个FastSAM实例
    std::unique_ptr<FastSam> fastsam_;
    std::vector<Object> objs_;
    std::vector<Segment> segments_;
    int cnt_;
    double time_sum_;
    cv::Mat color_mask_, image_;
    cv::Size fix_size_; // 指的是网络固定的输入大小
};

}