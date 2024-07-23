#pragma once
#include "clip.cpp/clip.h"
#include "clip_instance/common-clip.h"
#include <string>

// Clip单例类，实际上是对clip.cpp的再封装，用于加载Clip模型，接收mask类型输入，然后输出该mask的特征向量

struct Mask {
    int width;
    int height;
    // 所谓的mask，实际上是通过sam生成的掩码
};

struct CropImg : clip_image_u8 {
    // 通过mask裁剪得到的crop img
    Mask mask;
    

};

namespace SamToClip {
class ClipInstance {
    // clip 参数
public:
    ClipInstance(std::string yaml_file);
    ~ClipInstance();
    bool load_model();
    bool generate_mask_feature(CropImg& crop_img);
    bool generate_masks_feature(std::vector<CropImg>& crop_imgs);

    bool generate_text_feature();


private:
    app_params params_;
    clip_ctx* clip_ctx_;
    int64_t time_infer_sum_;
    int cnt_;
    

};
}


