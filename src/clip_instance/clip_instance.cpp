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

    embedding_dim_ = clip_get_vision_hparams(clip_ctx_)->projection_dim;
}

ClipInstance::~ClipInstance() {
    clip_free(clip_ctx_);
}

// generate_text_feature
float* ClipInstance::generate_text_feature(const char* text) {
    const int vec_dim = embedding_dim_;
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

bool ClipInstance::generate_segments_feature(std::vector<Segment*>& segments) {
    // 以batch的方式
    if (segments.size() == 0) {
        std::cout << "Segments' size is zero!" << std::endl;
        return false;
    }
    std::cout << "Generating segments feature..." << std::endl;
    auto start_us = ggml_time_us();

    const int vec_dim = embedding_dim_;
    float vec[vec_dim];

    for (Segment* seg : segments) {
        clip_image_f32 img_res;
        if (!to_clip_image_f32(img_res, seg->img)) {
            std::cout << "Unable to get clip image f32" << std::endl;
            continue;
        } else {
            // feature
            clip_image_encode(clip_ctx_, params_.n_threads, &img_res, vec, true); // 不同大小估计不能用clip_image_encode_batch
            seg->embedding_dim = vec_dim;
            seg->embedding = new float[vec_dim]();
            memcpy(seg->embedding, vec, vec_dim * sizeof(float));
            checkNorm(vec, vec_dim);
        }
    }
    auto end_us = ggml_time_us();

    if (params_.verbose >= 1) {
        printf("\n%s: Generated segments embedding in  %8.2f ms\n", __func__, (end_us - start_us) / 1000.0);
    }
    return true;
}

bool ClipInstance::generate_segments_feature_batch(std::vector<Segment*>& segments) {
    // 以batch的方式，仍然存在一些问题，先不用batch推理吧
    if (segments.size() == 0) {
        std::cout << "Segments' size is zero!" << std::endl;
        return false;
    }
    std::cout << "Generating segments feature..." << std::endl;
    auto start_us = ggml_time_us();

    const int vec_dim = embedding_dim_;
    
    // batch 不能设置太大，当作一个参数吧
    const size_t batch_size = 4;
    std::vector<clip_image_u8> u8_batch_data(batch_size);
    std::vector<Segment*> valid_segments(batch_size);
    std::vector<clip_image_f32> f32_batch_data(batch_size);

    auto u8_batch = clip_image_u8_batch{};
    auto f32_batch = clip_image_f32_batch{};

    f32_batch.data = f32_batch_data.data();
    f32_batch.size = f32_batch_data.size();

    for (size_t i = 0; i < segments.size(); i += batch_size) {
        // 同一个batch
        u8_batch_data.clear();
        valid_segments.clear();

        
        for (int j = 0; j < batch_size; j++) {
            clip_image_u8 img_u8;
            Segment* seg = segments[i + j];
            if (i + j < segments.size()) {
                if (seg->valid && to_clip_image_u8(img_u8, seg->img)) {
                    // valid_segments[j] = seg; // 有可能部分为空的！所以还是要clear
                    // u8_batch_data[j] = img_u8;
                    valid_segments.push_back(seg);
                    u8_batch_data.push_back(img_u8);
                } else {
                    std::cout << "Unable to get clip image u8 in generate_segments_feature_batch" << std::endl;
                }
            }
        }

        u8_batch.data = u8_batch_data.data(); // 共享同一份内存
        u8_batch.size = u8_batch_data.size();

        // 然后调用clip_image_batch_preprocess 获取clip_image_f32_batch

        clip_image_batch_preprocess(clip_ctx_, params_.n_threads, &u8_batch, &f32_batch);
        // 调用clip_image_batch_encode得到 float * vec;
        float* vec = new float[vec_dim * u8_batch.size]();
        clip_image_batch_encode(clip_ctx_, params_.n_threads, &f32_batch, vec, true);

        // 通过vec_dim偏移得到每个seg的embedding
        for (size_t k = 0; k < valid_segments.size(); k++) {
            Segment* seg = valid_segments[k];
            seg->embedding_dim = vec_dim;
            seg->embedding = new float[vec_dim]();
            memcpy(seg->embedding, vec + k * vec_dim, vec_dim * sizeof(float));
        }
    }
    auto end_us = ggml_time_us();

    if (params_.verbose >= 1) {
        printf("\n%s: Generated segments embedding use batch in  %8.2f ms\n", __func__, (end_us - start_us) / 1000.0);
    }

    return true;
}

// common function
std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    float maxVal = *std::max_element(input.begin(), input.end());
    
    float sum = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - maxVal); // 减去最大值避免溢出
        sum += output[i];
    }
    
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= sum; // 归一化
    }
    
    return output;
}



// Image数据结构
// 当前帧
// 传入 当前帧image，Segments
// 根据image的clip特征更新segments的embedding
bool ClipInstance::update_segments_feature(const cv::Mat& img, std::vector<Segment*>& segments) {
    clip_image_f32 img_res;
    if (!to_clip_image_f32(img_res, img)) {
        std::cout << "Unable to get clip image f32" << std::endl;
        return false;
    } else {
        // feature
        float* global_embedding = new float[embedding_dim_];
        clip_image_encode(clip_ctx_, params_.n_threads, &img_res, global_embedding, true);
        
        // 计算原图 feature 和各个segment的feature的余弦相似度
        // TODO(通过矩阵进行运算)
        std::vector<float> weight_l_g(segments.size());
        for (int i = 0; i < segments.size(); i++) {
            weight_l_g[i] = clip_similarity_score(global_embedding, segments[i]->embedding, embedding_dim_);
        }

        weight_l_g = softmax(weight_l_g);
        
        float* norm = new float[segments.size()];
        // 根据权重更新segment的embedding
        for (int i = 0; i < segments.size(); i++) {
            for (int k = 0; k < embedding_dim_; k++) {
                segments[i]->embedding[k] = segments[i]->embedding[k] * (1 - weight_l_g[i]) + global_embedding[k] * weight_l_g[i];
                norm[i] += pow(segments[i]->embedding[k], 2);
            }
        }

        // normalize
        for (int i = 0; i < segments.size(); i++) {
            for (int k = 0; k < embedding_dim_; k++) {
                segments[i]->embedding[k] /= sqrt(norm[i]);
            }
        }

        return true;
    }
}



void ClipInstance::compute_image_text_similarity(clip_image_u8& res, const char* text) {
    float score;
    clip_compare_text_and_image(clip_ctx_, params_.n_threads, text, &res, &score);
    std::cout << "score with "<< text << " is " << score << std::endl;
}


bool ClipInstance::to_clip_image_u8(clip_image_u8& res, const cv::Mat& img) {
    // transfer to clip_image_u8
    // cv::Mat is BGR!!!
    // stbi is RGBRGBRGB
    if (!img.data) {
        return false;
    }

    cv::Mat image_RGB = img.clone();
    cv::cvtColor(img, image_RGB, cv::COLOR_BGR2RGB);


    res.nx = img.cols;
    res.ny = img.rows;
    res.size = img.cols * img.rows * 3;
    res.data = new uint8_t[res.size]();

    memcpy(res.data, image_RGB.data, res.size * sizeof(uint8_t));
    
    return true;
}

bool ClipInstance::to_clip_image_f32(clip_image_f32& res, const cv::Mat& img) {
    // transfer to clip_image_u8 and preprocess to clip_image_f32
    // cv::Mat is BGR!!!
    // stbi is RGBRGBRGB
    if (!img.data) {
        return false;
    }
    cv::Mat image_RGB;
    cv::cvtColor(img, image_RGB, cv::COLOR_BGR2RGB);

    clip_image_u8 temp;

    temp.nx = img.cols;
    temp.ny = img.rows;
    temp.size = img.cols * img.rows * 3;
    temp.data = new uint8_t[temp.size]();
    memcpy(temp.data, image_RGB.data, temp.size * sizeof(uint8_t));
    
    if (!clip_image_preprocess(clip_ctx_, &temp, &res)) {
        std::cout << "Unable to preprocess image" << std::endl;
    }
    
    return true;
}

}