#include "SegmentDetector.h"
#include <yaml-cpp/yaml.h>
#include <map>

using namespace std;
using namespace perception;
using namespace camera;

SegmentDetector::SegmentDetector()
{

}
SegmentDetector::~SegmentDetector()
{

}

bool SegmentDetector::init(const string &config_path)
{
    YAML::Node config = YAML::LoadFile(config_path);
    Option option;
    option.weight = config["ddrnet"]["weight"].as<string>();
    option.precison = config["ddrnet"]["precison"].as<string>();
    option.mean_value = config["ddrnet"]["mean"].as<vector<float>>();
    option.std_value = config["ddrnet"]["std"].as<vector<float>>();

    if(!ddrnet.init(option))
        return false;
    return true;
}

cv::Mat SegmentDetector::proc(cv::Mat &frame)
{
    cv::Mat img = frame.clone();
    cv::Mat gary = ddrnet.segment(img);
    cv::Mat result = ddrnet.ColorMap(gary);

    return result;
}