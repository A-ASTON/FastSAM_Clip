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
    image_ = img.clone(); // 执行推理后缓存了image
    auto start = std::chrono::system_clock::now();
    fastsam_->run(image_, objs_);
    auto end = std::chrono::system_clock::now();
    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(
                end - start)
                .count() /
                1000.;
    printf("cost %2.4lf ms\n", tc);
    ++cnt_;
    time_sum_ += tc;
    printf("avg cost %2.4lf ms\n", time_sum_ / cnt_);
    fastsam_->draw_objects(image_, color_mask_, objs_);
    cv::imshow("result", color_mask_);
    cv::waitKey(0);
}

void FastSamInstance::crop_images(bool filter) {
    //利用objs_更新crop_images_
    // 不过滤
    std::vector<int> filter_id;
    segments_.clear();

    int id = 0;
    for (auto& obj : objs_) {
        Segment seg = segment_image(obj, id);
        if (!filter) {
            segments_.push_back(seg); // 根据obj进行crop，然后压入segments_
        } else {
            // 过滤掉area太小的区域
            if (seg.area < 500) {
                filter_id.push_back(id);
            } else {
                segments_.push_back(seg);
            }
        }
        id++;
        
    }
}

Segment FastSamInstance::segment_image(const Object& obj, const int id) {
    // 利用内部image_即原图进行crop
    Segment seg;
    seg.id = id;
    seg.bbox = obj.rect;
    seg.score = obj.prob;
    // 利用obj.boxMask 进行crop
    if (image_.empty()) {
        std::cout << "Error: Image is NULL!" << std::endl; 
        return seg;
    }
    cv::Mat crop_image = image_.clone();
    // 获取最小外接矩阵，然后得到对应的掩码

    // 根据mask和bbox将目标区域保留,获取原图大小的mask
    cv::Mat mask = cv::Mat::zeros(image_.size(), CV_8UC1); // 代表保留原图大小，只有目标object mask
    mask(seg.bbox).setTo(cv::Scalar(255), obj.boxMask);
    
    std::vector<std::vector<cv::Point>> contours; // 存储轮廓坐标
    cv::Mat hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 根据轮廓，找到最小外接矩形，为什么不直接用bbox呢？
    // 方法一：直接保留原图大小的数据
    // 方法二：最小外接矩阵
    // 如何取图影响clip性能
    cv::Rect rect = cv::boundingRect(contours[0]);
    int x1 = rect.x;
    int y1 = rect.y;
    int w = rect.width;
    int h = rect.height;
    int x2 = x1 + w;
    int y2 = y1 + h;
    if (contours.size() > 1) {
        // 多个轮廓，找最外层轮廓
        for (size_t i = 1; i < contours.size(); i++) {
            rect = cv::boundingRect(contours[i]);
            x1 = std::min(x1, rect.x);
            y1 = std::min(y1, rect.y);
            x2 = std::max(x2, rect.x + rect.width);
            y2 = std::max(y2, rect.y + rect.height);
        }
        w = x2 - x1;
        h = y2 - y1;
    }
    // 最小外接矩阵掩码
    cv::Mat new_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    clamp(x1, 0, mask.rows);
    clamp(x2, 0, mask.rows);
    clamp(y1, 0, mask.cols);
    clamp(y2, 0, mask.cols);

    cv::Mat crop_mask = cv::Mat::zeros(image_.size(), CV_8UC1);

    cv::rectangle(new_mask, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), -1); // 掩码填充

    // 这一部分相当于方法一, FastSAM中使用的方法
    // 反转掩码 
    cv::bitwise_not(new_mask, new_mask);

    // 把反转掩码的区域设置为白色
    cv::Scalar bg_color = {255, 255, 255};
    // 保留掩码处的图像
    // cv::resize(crop_image, crop_image, newMask.size());
    crop_image.setTo(bg_color, new_mask); 
    seg.img = crop_image;
    seg.area = new_mask.size().area();

    // cv::imshow("crop_image", crop_image);
    // cv::waitKey(0);
    // 

    // 方法二，得到裁剪出来的图像
    // cv::Mat crop = image_(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    // seg.img = crop;
    // seg.area = crop.size().area();

    // cv::imshow("crop_image", crop);
    // cv::waitKey(0);

    return seg;
}

std::vector<Segment> FastSamInstance::get_segments() {
    return segments_;
}


}