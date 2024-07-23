#include "clip_instance/clip_instance.h"


namespace SamToClip {

ClipInstance::ClipInstance(std::string yaml_path) {
    printf("Loading model");
    app_params_yaml(yaml_path, params_);
    printf("Generated app_params");
    // 加载clip_model
    clip_ctx_ = clip_model_load(params_.model.c_str(), params_.verbose); 
    if (!clip_ctx_) {
        printf("%s: Unable to load model from %s", __func__, params_.model.c_str());
        exit(0); // 无法加载clip，直接退出？
    } else {
        printf("Loaded CLIP model %s\n", params_.model.c_str());
    }
}

ClipInstance::~ClipInstance() {
    clip_free(clip_ctx_);
}

bool ClipInstance::load_model() {
    return true;
}


bool generate_mask_feature(CropImg& crop_img) {
    // 生成单个mask的feature
}
bool generate_masks_feature(std::vector<CropImg>& crop_imgs) {
    // 生成多个mask的feature，通常属于一张image
}


}