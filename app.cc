#include "SegmentDetector.h"
#include <vector>

using namespace std;

int main()
{
    std::string config_path = "/home/bzl/Documents/rk_seg/seg_tensorrt/config.yaml";
    SegmentDetector ddrnet;
    ddrnet.init(config_path);

    //检测图片....................................
    // cv::Mat image = cv::imread("/home/bzl/Documents/segment/images/00072.jpg");
    // cv::Mat result = ddrnet.proc(image);
    // cv::imshow("segggg", result);
    // cv::waitKey(0);

    //.............................................

    // //检测视频......................................
    cv::VideoCapture cap("/home/bzl/Documents/data/data/video/darknet_test_vedio/2022-04-15-10-00-05.avi");

    cv::Mat frame;
    while (cap.read(frame))
    {
        cv::Mat result = ddrnet.proc(frame);
        cv::addWeighted(frame, 0.7, result, 0.3, 1, result);
        cv::imshow("live", result);
        cv::waitKey(1);
    }
    // //.............................................

    return 0;
}