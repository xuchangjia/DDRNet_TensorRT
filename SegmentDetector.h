#pragma once
#include <string>
#include "ddrnet/DDRNet.h"

class SegmentDetector
{
private:
    perception::camera::DDRNet ddrnet;

public:
    SegmentDetector();
    ~SegmentDetector();
    bool init(const std::string &config_path);
    cv::Mat proc(cv::Mat &frame);
};
