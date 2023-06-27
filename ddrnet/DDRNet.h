//
// Created by qin on 2022/7/14.
//
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <NvInfer.h>
#include "TRTLogger.hpp"

namespace perception {
namespace camera {
    struct Option
    {
        std::string weight;
        std::string calib;
        std::string precison;
        std::vector<float> mean_value;
        std::vector<float> std_value;
    };

    class DDRNet {
    private:
        TRTLogger logger_;
        Option option_;
        std::string engine_file_;
        float* input_;
        float* output_;
        nvinfer1::ICudaEngine* engine_;
    public:
        DDRNet(/* args */);
        ~DDRNet();
        bool init(Option&);
        cv::Mat segment(cv::Mat&);
        cv::Mat ColorMap(cv::Mat&);

    private:
        bool LoadEngine();
        bool BuildEngine();
        cv::Mat ParseOutput();
        std::map<std::string, nvinfer1::Weights> LoadWeights();
        void cvimage2input(cv::Mat&);
    };
}//namespace camera
}//namespace perception

