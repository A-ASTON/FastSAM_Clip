#include "clip_instance/clip_instance.h"


namespace SamToClip {

ClipInstance::ClipInstance(std::string yaml_path) {
    ggml_time_init();

    app_params_yaml(yaml_path, params_);
    std::cout << "Generated app_params." << std::endl;

    std::cout << "Loading Clip Model..." << std::endl;
    const int64_t t_load_us = ggml_time_us();
    // 加载clip_model
    clip_ctx_ = clip_model_load(params_.model.c_str(), params_.verbose); 
    if (!clip_ctx_) {
        printf("%s: Unable to load model from %s", __func__, params_.model.c_str());
        exit(0); // 无法加载clip，直接退出？
    } else {
        printf("Loaded CLIP model %s\n", params_.model.c_str());
    }
    const int64_t t_load_end_us = ggml_time_us();

    if (params_.verbose >= 1) {
        printf("\n\nTimings\n");
        printf("%s: Model loaded in %8.2f ms\n", __func__, (t_load_end_us - t_load_us) / 1000.0);
    }
}

ClipInstance::~ClipInstance() {
    clip_free(clip_ctx_);
}

bool ClipInstance::load_model() {
    return true;
}

// generate_text_feature
float* ClipInstance::generate_text_feature(const char* text) {
    const int vec_dim = clip_get_vision_hparams(clip_ctx_)->projection_dim;
    float* vec = new float[vec_dim]();

    std::cout << "Generating text feature..." << std::endl;
    auto start_us = ggml_time_us();

    clip_tokens tokens;
    if (!clip_tokenize(clip_ctx_, text, &tokens)) {
        printf("Tokenize failed\n");
        return nullptr;
    }
    clip_text_encode(clip_ctx_, params_.n_threads, &tokens, vec, true);

    auto end_us = ggml_time_us();

    if (params_.verbose >= 1) {
        printf("\n%s: Generated text embedding in  %8.2f ms\n", __func__, (end_us - start_us) / 1000.0);
    }
    return vec;
}

bool ClipInstance::generate_segments_feature(std::vector<Segment>& segments) {
    if (segments.size() == 0) {
        std::cout << "Segments' size is zero!" << std::endl;
        return false;
    }
    std::cout << "Generating segments feature..." << std::endl;
    auto start_us = ggml_time_us();

    const int vec_dim = clip_get_vision_hparams(clip_ctx_)->projection_dim;
    float vec[vec_dim];

    for (Segment& seg : segments) {
        clip_image_f32 img_res;
        if (!to_clip_image_f32(img_res, seg.img)) {
            std::cout << "Unable to get clip image f32" << std::endl;
            continue;
        } else {
            // feature
            clip_image_encode(clip_ctx_, params_.n_threads, &img_res, vec, true); // 不同大小估计不能用clip_image_encode_batch
            seg.embedding_dim = vec_dim;
            seg.embedding = new float[vec_dim]();
            memcpy(seg.embedding, vec, vec_dim * sizeof(float));
        }
    }
    auto end_us = ggml_time_us();

    if (params_.verbose >= 1) {
        printf("\n%s: Generated segments embedding in  %8.2f ms\n", __func__, (end_us - start_us) / 1000.0);
    }
    return true;
}

bool ClipInstance::generate_mask_feature(Segment& seg) {
    // 生成单个mask的feature
    clip_image_f32 img_res;
    to_clip_image_f32(img_res, seg.img);

    const int vec_dim = clip_get_vision_hparams(clip_ctx_)->projection_dim;
    int shape[2] = {1, vec_dim};
    float vec[vec_dim]; // feature
    clip_image_encode(clip_ctx_, params_.n_threads, &img_res, vec, false);
    
}

void ClipInstance::compute_image_text_similarity(clip_image_u8& res, const char* text) {
    // clip_compare_text_and_image(const clip_ctx * ctx, const int n_threads, const char * text, const clip_image_u8 * image,
    //                              float * score)
    float score;
    clip_compare_text_and_image(clip_ctx_, params_.n_threads, text, &res, &score);
    std::cout << "score with "<< text << " is " << score << std::endl;
}
// bool generate_masks_feature(std::vector<CropImg>& crop_imgs) {
//     // 生成多个mask的feature，通常属于一张image
// }
 
bool ClipInstance::to_clip_image_f32(clip_image_f32& res, const cv::Mat& img) {
    // transfer to clip_image_u8
    if (!img.data) {
        return false;
    }

    clip_image_u8 temp;

    temp.nx = img.cols;
    temp.ny = img.rows;
    temp.size = img.cols * img.rows;
    // temp.data = static_cast<uint8_t*>(img.data); // 只是将指针的值进行复制，并没有进行内存的复制
    temp.data = new uint8_t[temp.size]();
    memcpy(temp.data, img.data, temp.size);
    
    if (!clip_image_preprocess(clip_ctx_, &temp, &res)) {
        std::cout << "Unable to preprocess image" << std::endl;
    }
    
    return true;
}

}