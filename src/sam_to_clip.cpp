#include <iostream>
#include "clip.cpp/clip.h"
#include "clip_instance/clip_instance.h"
#include "fastsam_instance/fastam_instance.h"
#include "common.h"

int main(int argc, char *argv[]) {
    std::string file("/home/catkin_ws/projects/clip_sam_ws/SamToClip/config/clip.yaml");
    SamToClip::ClipInstance clip_instance(file);

    std::string engine_file("/home/catkin_ws/projects/clip_sam_ws/SamToClip/models/FastSAM.engine");
    SamToClip::FastSamInstance fastsam_instance(engine_file);

    cv::Mat img = cv::imread(clip_instance.params_.image_paths[0]);
    fastsam_instance.infer(img);
    // 得到的objs包含了目标框和物体的mask
    fastsam_instance.crop_images(true);

    std::vector<SamToClip::Segment> segments = fastsam_instance.get_segments();
    clip_instance.generate_segments_feature(segments);

    const char *text = clip_instance.params_.texts[0].c_str();
    float *vec = clip_instance.generate_text_feature(text);  
    
    float max_score = 0;
    int max_idx = 0;
    int idx = 0;

    for (auto& seg : segments) {
        if (seg.embedding_dim != 0) {
            // std::cout << " seg.embedding_dim: " << seg.embedding_dim << std::endl;
            float similarity_score = clip_similarity_score(vec, seg.embedding, seg.embedding_dim);
            if (similarity_score > max_score) {
                max_score = similarity_score;
                max_idx = idx;
            }
        }
        idx++;
    } 

    std::cout << "max_score: " << max_score << std::endl;
    cv::imshow("Highly Realted to text " + std::string(text), segments[max_idx].img);
    cv::waitKey(0);


    // 处理逻辑
    // fastsam_instance.format_results(); // 处理为一个个对象的形式，参照FastSAM

    // fastsam_instance.crop_image(); // 根据format_result，对image进行crop，得到crop image

    // clip_instance.generate_masks_feature(); //计算每个crop image的embedding

    // clip_instance.generate_texts_feature(); // 这个可以在初始化阶段进行计算，并存储到内存中，减少开销

    // clip_instance.similarity_compute(); // 计算每个crop和各个text之间的相似度得分

    // clip_instance.generate_masks_feature(fastsam_instance.objs_);
    // object

    // 如何将单独的object，得到crop image，然后送入clip？

    // 读取一张image，然后调用SAM获取Mask List

    // 将Mask List使用CLIP进行批量处理，得到一张图片的上面的Mask的Embedding

    // Mask作为一个struct，包含Mask的Embedding，以及Mask的Bounding Box，Image 新增Masks和Embedding
    // 此外，增加一个单例类，加载CLIP模型，缓存CLIP模型，然后接收Mask并返回修改后的Mask

    return 0;
}


