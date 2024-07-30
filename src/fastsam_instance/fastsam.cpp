
#include "fastsam_instance/fastsam.h"

/**
 * @brief 依据长边等比例缩放,短边部分填充
 * @param image 输入图像
 * @param out 输出图像
 * @param size 缩放至尺寸
 */
void FastSam::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size)
{
    const float inp_h = size.height; // target size 1024
    const float inp_w = size.width;
    float height = image.rows; // img size
    float width = image.cols;

    // ratio最小值，说明以长边进行缩放，因为分母越大，数越小
    // 将长边变换为目标尺寸
    // 短板根据长边的变换比例进行
    float r = std::min(inp_h / height, inp_w / width); // ratio 比如 480 * 640, 1024/480 = 2.13333 , 1024/640 = 1.6
    int padw = std::round(width * r); // padw = 640 * 1.6 = 1024
    int padh = std::round(height * r); // padh = 480 * 1.6 = 768

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh)
    {
        cv::resize(image, tmp, cv::Size(padw, padh)); // resize进行缩放
    }
    else
    {
        tmp = image.clone();
    }

    float dw = inp_w - padw; // 0
    float dh = inp_h - padh; // 1024 - 768 = 256

    dw /= 2.0f; // 0
    dh /= 2.0f; // 128
    int top = int(std::round(dh - 0.1f)); // 128
    int bottom = int(std::round(dh + 0.1f)); // 128
    int left = int(std::round(dw - 0.1f)); // 0
    int right = int(std::round(dw + 0.1f)); // 0

    // copyMakeBorder实现边界填充，top,bottom,left,right这四个参数的意思是在对应的图像边界外填充指定数目的像素
    // 比如top=128，bottom=128表明在上边界添加128pixel 高的像素带
    // 目的是将图像填充到1024 * 1024
    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT,
                       {0, 0, 0});
    

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0),
                           true, false, CV_32F);
    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;
}

/**
 * @brief 输入图像预处理,序列化,转移至显存
 * @param image
 */
void FastSam::copy_from_Mat(const cv::Mat &image)
{
    cv::Mat nchw;
    auto &in_binding = this->input_bindings[0];
    auto width = in_binding.dims.d[3];
    auto height = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    this->context->setBindingDimensions(0,
                                        nvinfer1::Dims{4, {1, 3, height, width}}); // 设置输入维度，一个具有四个维度的张量，尺寸为(1,3,H,W)

    // 将数据拷贝到设备上
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(),
                          nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice,
                          this->stream));
}

/**
 * @brief 输入图像预处理,序列化,转移至显存
 * @param image
 * @param size 指定输入尺寸
 */
void FastSam::copy_from_Mat(const cv::Mat &image, cv::Size &size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(
        0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(),
                          nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice,
                          this->stream));
}

/**
 * @brief 分割解码后处理
 * @param objs 所有对象数组
 */
void FastSam::postprocess(std::vector<Object> &objs)
{
    objs.clear();
    auto input_h = this->input_bindings[0].dims.d[2]; // 通过input_bindings和output_bindings获取维度信息
    auto input_w = this->input_bindings[0].dims.d[3]; // 1024
    int mask_channel = this->output_bindings[0].dims.d[1]; // 32
    int mask_h = this->output_bindings[0].dims.d[2]; //256
    int mask_w = this->output_bindings[0].dims.d[3]; //256
    int class_number = 1; // 类别数，这是类别无关分割
    int mask_number = this->output_bindings[5].dims.d[1] - class_number - 4; // (37 - class_number 1 - bbox 4) = 32
    int candidates = this->output_bindings[5].dims.d[2]; // 21504
    // this->output_bindings[5].dims.d 1*37*21504
    // 37 = 4+1+32 box+conf+mask

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> mask_confs;
    std::vector<int> indices;

    auto &dw = this->pparam.dw; // 宽度变化量 width 
    auto &dh = this->pparam.dh;
    auto &width = this->pparam.width; // 原始宽度
    auto &height = this->pparam.height; // 原始高度
    auto &ratio = this->pparam.ratio; // 1 / r也就是 原始尺寸/固定尺寸

    // 通过host_ptrs获取数据，host_ptrs的数据来源于device_ptrs
    auto *output = static_cast<float *>(this->host_ptrs[5]); // 输出 debug到此处
    cv::Mat protos = cv::Mat(mask_channel, mask_h * mask_w, CV_32F,
                             static_cast<float *>(this->host_ptrs[0])); // mask_channel, mask_h*mask_w, 32f
    // 遍历每一个最后一维的数据
    // 数据存储形式应该是这样的
    // d0,0 d0,1 d0,2
    // 说白了就是一个二维数组嘛，但是用一维数组存放，所以
    // a[i][j] = a[i * j.dim + j]
    // 现在，遍历j，然后取出对应每个i
    // 这里i就是j，j.dim就是candidates，i就是对应的维度嘛！

    for (size_t i = 0; i < candidates; ++i) // 1, 37, 21504, candidates是21504
    {
        float score = *(output + 4 * candidates + i); // score 相当于 output[4][i]
        if (score > _score_thres)
        {
            // center_x, center_y, width, height
            float w = *(output + 2 * candidates + i); // w 相当于 output[2][i]
            float h = *(output + 3 * candidates + i); // h 相当于 output[3][i]

            float x0 = *(output + 0 * candidates + i) - dw - w / 2; // x0 相当于 output[0][i], 这里将中心点x0， y0变换到原图，然后分别减去w/2和h/2
            float y0 = *(output + 1 * candidates + i) - dh - h / 2; // y0 相当于 output[1][i]

            float x1 = x0 + w; // 右上角x
            float y1 = y0 + h; // 右上角y

            x0 = clamp(x0 * ratio, 0.f, width); // 限制在原始尺寸内
            y0 = clamp(y0 * ratio, 0.f, height);
            x1 = clamp(x1 * ratio, 0.f, width);
            y1 = clamp(y1 * ratio, 0.f, height);
            // 宽/高为0的直接舍弃，以免后续取整后rect roi为空矩阵
            if (x1 - x0 < 1 || y1 - y0 < 1)
                continue;
            float *mask_conf = new float[mask_number]; // mask置信度
            for (size_t j = 0; j < mask_number; ++j)
            {
                mask_conf[j] = *(output + (5 + j) * candidates + i); // 就是那32维的数据
            }

            cv::Mat mask_conf_mat = cv::Mat(1, mask_number, CV_32F, mask_conf); // mask_conf初始化该矩阵
            mask_confs.push_back(mask_conf_mat);
            // labels.push_back(label);
            scores.push_back(score);
            // 传入的应该是左上角才对？x0, y0是左下角
            // No! x0, y0 是左上角，因为坐标原点在左上角
            // x向右增加，y向下增加
            // x越小越左 y越小越上
            bboxes.push_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0)); 
        }
    }
    cv::dnn::NMSBoxes(bboxes, scores, _score_thres, _iou_thres, indices);
    cv::Mat masks;
    int cnt = 0;
    // 通过NMSBoxes处理后剩下的数据
    for (auto &i : indices)
    {
        if (cnt >= _topk)
        {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object obj;
        // obj.label = labels[i];
        obj.rect = tmp;
        obj.prob = scores[i];
        masks.push_back(mask_confs[i]);
        objs.push_back(obj);
        cnt += 1;
    }
    if (masks.empty())
    {
        // masks is empty
    }
    else
    {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(), {mask_w, mask_h});

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        int scale_dw = dw / input_w * mask_w;
        int scale_dh = dh / input_h * mask_h;

        cv::Rect roi(scale_dw, scale_dh, mask_w - 2 * scale_dw,
                     mask_h - 2 * scale_dh);

        for (int i = 0; i < indices.size(); i++)
        {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi); 
            cv::resize(dest, mask, cv::Size((int)width, (int)height),
                       cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].rect) > _mask_conf;
        }
    }
}

/**
 * @brief 可视化
 * @param image
 * @param res
 * @param objs
 */
void FastSam::draw_objects(const Mat &image, Mat &res, const vector<Object> &objs)
{
    res = image.clone(); // 原图的copy
    Mat color_mask = image.clone(); // 原图的copy
    std::srand(std::time(0)); // 设置随机数种子
    // 根据area排序，先为面积大的区域着色
    vector<Object> objs_sorted = objs;
    std::sort(objs_sorted.begin(), objs_sorted.end(), [](const Object& obj1, const Object& obj2){
        return obj1.rect.area() > obj2.rect.area();   
    });
    for (auto &obj : objs_sorted)
    {
        // 生成随机的RGB值
        int r = std::rand() % 256;
        int g = std::rand() % 256;
        int b = std::rand() % 256;
        // if (obj.rect.area() < 20000)
        //     continue;
        cv::Scalar mask_color = {r, g, b};
        color_mask(obj.rect).setTo(mask_color, obj.boxMask);
        // res(obj.rect).setTo(mask_color, obj.boxMask);//obj.rect矩形区域内的obj.boxMask掩码部分附上mask_color
    }
    cv::addWeighted(res, 0.5, color_mask, 0.8, 1, res);
    // for (auto &obj : objs)
    // {   // 过滤小目标框
    //     if (obj.rect.area() < 20000)
    //         continue;
    //     cv::rectangle(res, obj.rect, cv::Scalar(0, 0, 255), 2);
    // }
}

FastSam::FastSam(const string &engine_file_path,const int warm_cnt) : Inference(engine_file_path)
{
    _input_size = cv::Size(1024, 1024);
    _score_thres = 0.25;
    _iou_thres = 0.5;
    _mask_conf = 0.5f;
    _topk = 100;
    this->make_pipe(warm_cnt,true);
}

void FastSam::run(const Mat &frame, vector<Object> &objs)
{
    this->copy_from_Mat(frame, _input_size); // 执行尺寸缩放并转移到显存中
    this->infer(); // 推理
    this->postprocess(objs); // 后处理
}
