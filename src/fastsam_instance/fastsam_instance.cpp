// 加载fastsam TRT模型并等候输入
#include "chrono"
#include "fastsam_instance/fastam_instance.h"
#include "fastsam_instance/common-fastsam.h"
#include <memory>


namespace SamToClip {
FastSamInstance::FastSamInstance(const string &engine_file_path) 
    : cnt_(0), time_sum_(0) {
    fastsam_ = std::make_unique<FastSam>(engine_file_path);
}

FastSamInstance::~FastSamInstance() {
    // unique_ptr会自动释放
}

void FastSamInstance::infer(const cv::Mat& img) {
    objs_.clear(); // 每次调用时，先清空结果列表
    image_ = img.clone();
    auto start = std::chrono::system_clock::now();
    fastsam_->run(image_, objs_);
    fastsam_->draw_objects(image_, color_mask_, objs_);
    auto end = std::chrono::system_clock::now();
    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(
                end - start)
                .count() /
                1000.;
    printf("cost %2.4lf ms\n", tc);
    ++cnt_;
    time_sum_ += tc;
    printf("avg cost %2.4lf ms\n", time_sum_ / cnt_);
    cv::imshow("result", color_mask_);
    cv::waitKey(0);
}

}