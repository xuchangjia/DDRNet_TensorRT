#pragma once

#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <chrono>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

namespace perception {
namespace camera {
namespace net {

using namespace nvinfer1;

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps)
{
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };
    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}


ILayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, bool downsample, bool no_relu, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ 3, 3 }, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ stride, stride });
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPaddingNd(DimsHW{ 1, 1 });
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);
    if(downsample)
    {
        IConvolutionLayer* convdown = network->addConvolutionNd(input, outch, DimsHW{ 1, 1 }, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(convdown);
        convdown->setStrideNd(DimsHW{ stride, stride});
        convdown->setPaddingNd(DimsHW{ 0, 0 });
        IScaleLayer* bndown = addBatchNorm2d(network, weightMap, *convdown->getOutput(0), lname + "downsample.1", 1e-5);
        IElementWiseLayer* ew1 = network->addElementWise(*bn2->getOutput(0), *bndown->getOutput(0), ElementWiseOperation::kSUM);
        if(no_relu){
            return ew1;
        }else{
            IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
            assert(relu3);
            return relu3;
        }
    }
    IElementWiseLayer* ew2 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    if(no_relu){
        return ew2;
    }else{
        IActivationLayer* relu3 = network->addActivation(*ew2->getOutput(0), ActivationType::kRELU);
        assert(relu1);
        return relu3;
    }
}

ILayer* Bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, bool downsample, bool no_relu, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ 1, 1 }, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ 1, 1 });
    conv1->setPaddingNd(DimsHW{ 0, 0 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ stride, stride });
    conv2->setPaddingNd(DimsHW{ 1, 1 });
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch*2, DimsHW{ 1, 1 }, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);
    conv3->setStrideNd(DimsHW{ 1, 1 });
    conv3->setPaddingNd(DimsHW{ 0, 0 });
    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);
    if(downsample)
    {
        IConvolutionLayer* convdown = network->addConvolutionNd(input, outch*2, DimsHW{ 1, 1 }, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(convdown);
        convdown->setStrideNd(DimsHW{ stride, stride });
        conv1->setPaddingNd(DimsHW{ 0, 0 });
        IScaleLayer* bndown = addBatchNorm2d(network, weightMap, *convdown->getOutput(0), lname + "downsample.1", 1e-5);
        IElementWiseLayer* ew1 = network->addElementWise(*bn3->getOutput(0), *bndown->getOutput(0), ElementWiseOperation::kSUM);
        if(no_relu){
            return ew1;
        }else{
            IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
            assert(relu1);
            return relu3;
        }
    }
    IElementWiseLayer* ew2 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    if(no_relu){
        return ew2;
    }else{
        IActivationLayer* relu3 = network->addActivation(*ew2->getOutput(0), ActivationType::kRELU);
        assert(relu1);
        return relu3;
    }
}


ILayer* compression3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int highres_planes, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, highres_planes , DimsHW{ 1, 1 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setPaddingNd(DimsHW{ 0, 0 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
    return bn1;
}

ILayer* compression4(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int highres_planes, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, highres_planes , DimsHW{ 1, 1 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setPaddingNd(DimsHW{ 0, 0 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
    return bn1;
}

ILayer* down3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int planes, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, planes , DimsHW{ 3, 3 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ 2, 2 });
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
    return bn1;
}

ILayer* down4(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int planes, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, planes , DimsHW{ 3, 3 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ 2, 2 });
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), planes*2 , DimsHW{ 3, 3 }, weightMap[lname + "3.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ 2, 2 });
    conv2->setPaddingNd(DimsHW{ 1, 1 });
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "4", 1e-5);
    return bn2;
}

ILayer* DAPPM(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inplanes, int branch_planes, int outplanes, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IScaleLayer* scale0bn = addBatchNorm2d(network, weightMap, input, lname + "scale0.0", 1e-5);
    IActivationLayer* scale0relu = network->addActivation(*scale0bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* scale0conv = network->addConvolutionNd(*scale0relu->getOutput(0), branch_planes , DimsHW{ 1, 1 }, weightMap[lname + "scale0.2.weight"], emptywts);
    assert(scale0conv);
    scale0conv->setPaddingNd(DimsHW{ 0, 0 });
    // x_list[1]
    IPoolingLayer* scale1pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{ 5, 5 });
    assert(scale1pool);
    scale1pool->setStrideNd(DimsHW{ 2, 2 });
    scale1pool->setPaddingNd(DimsHW{ 2, 2 });
    IScaleLayer* scale1bn = addBatchNorm2d(network, weightMap, *scale1pool->getOutput(0), lname + "scale1.1", 1e-5);
    IActivationLayer* scale1relu = network->addActivation(*scale1bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* scale1conv = network->addConvolutionNd(*scale1relu->getOutput(0), branch_planes , DimsHW{ 1, 1 }, weightMap[lname + "scale1.3.weight"], emptywts);
    assert(scale1conv);
    scale1conv->setPaddingNd(DimsHW{ 0, 0 });
    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * branch_planes * 2 * 2));
    for (int i = 0; i < branch_planes * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts1{ DataType::kFLOAT, deval, branch_planes * 2 * 2 };
    IDeconvolutionLayer* scale1_interpolate = network->addDeconvolutionNd(*scale1conv->getOutput(0), branch_planes, DimsHW{ 2, 2 }, deconvwts1, emptywts);
    scale1_interpolate->setStrideNd(DimsHW{ 2, 2 });
    scale1_interpolate->setNbGroups(branch_planes);
    IElementWiseLayer* process1_input = network->addElementWise(*scale1_interpolate->getOutput(0), *scale0conv->getOutput(0), ElementWiseOperation::kSUM);
    IScaleLayer* process1bn = addBatchNorm2d(network, weightMap, *process1_input->getOutput(0), lname + "process1.0", 1e-5);
    IActivationLayer* process1relu = network->addActivation(*process1bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* process1conv = network->addConvolutionNd(*process1relu->getOutput(0), branch_planes , DimsHW{ 3, 3 }, weightMap[lname + "process1.2.weight"], emptywts);
    assert(process1conv);
    process1conv->setPaddingNd(DimsHW{ 1, 1 });
    // x_list[2]
    IPoolingLayer* scale2pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{ 9, 9 });
    assert(scale2pool);
    scale2pool->setStrideNd(DimsHW{ 4, 4 });
    scale2pool->setPaddingNd(DimsHW{ 4, 4 });
    IScaleLayer* scale2bn = addBatchNorm2d(network, weightMap, *scale2pool->getOutput(0), lname + "scale2.1", 1e-5);
    IActivationLayer* scale2relu = network->addActivation(*scale2bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* scale2conv = network->addConvolutionNd(*scale2relu->getOutput(0), branch_planes , DimsHW{ 1, 1 }, weightMap[lname + "scale2.3.weight"], emptywts);
    assert(scale2conv);
    scale2conv->setPaddingNd(DimsHW{ 0, 0 });
    float *deval2 = reinterpret_cast<float*>(malloc(sizeof(float) * branch_planes * 4 * 4));
    for (int i = 0; i < branch_planes * 4 * 4; i++) {
        deval2[i] = 1.0;
    }
    Weights deconvwts2{ DataType::kFLOAT, deval2, branch_planes * 4 * 4 };
    IDeconvolutionLayer* scale2_interpolate = network->addDeconvolutionNd(*scale2conv->getOutput(0), branch_planes, DimsHW{ 4, 4 }, deconvwts2, emptywts);
    scale2_interpolate->setStrideNd(DimsHW{ 4, 4 });
    scale2_interpolate->setNbGroups(branch_planes);
    IElementWiseLayer* process2_input = network->addElementWise(*scale2_interpolate->getOutput(0), *process1conv->getOutput(0), ElementWiseOperation::kSUM);
    //  process2
    IScaleLayer* process2bn = addBatchNorm2d(network, weightMap, *process2_input->getOutput(0), lname + "process2.0", 1e-5);
    IActivationLayer* process2relu = network->addActivation(*process2bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* process2conv = network->addConvolutionNd(*process2relu->getOutput(0), branch_planes , DimsHW{ 3, 3 }, weightMap[lname + "process2.2.weight"], emptywts);
    assert(process2conv);
    process2conv->setPaddingNd(DimsHW{ 1, 1 });
    // scale3
    IPoolingLayer* scale3pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{ 17, 17 });
    assert(scale3pool);
    scale3pool->setStrideNd(DimsHW{ 8, 8 });
    scale3pool->setPaddingNd(DimsHW{ 8, 8 });
    IScaleLayer* scale3bn = addBatchNorm2d(network, weightMap, *scale3pool->getOutput(0), lname + "scale3.1", 1e-5);
    IActivationLayer* scale3relu = network->addActivation(*scale3bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* scale3conv = network->addConvolutionNd(*scale3relu->getOutput(0), branch_planes , DimsHW{ 1, 1 }, weightMap[lname + "scale3.3.weight"], emptywts);
    assert(scale3conv);
    scale3conv->setPaddingNd(DimsHW{ 0, 0 });
    float *deval3 = reinterpret_cast<float*>(malloc(sizeof(float) * branch_planes * 8 * 8));
    for (int i = 0; i < branch_planes * 8 * 8; i++) {
        deval3[i] = 1.0;
    }
    Weights deconvwts3{ DataType::kFLOAT, deval3, branch_planes * 8 * 8 };
    IDeconvolutionLayer* scale3_interpolate = network->addDeconvolutionNd(*scale3conv->getOutput(0), branch_planes, DimsHW{ 8, 8 }, deconvwts3, emptywts);
    scale3_interpolate->setStrideNd(DimsHW{ 8, 8 });
    scale3_interpolate->setNbGroups(branch_planes);
    IElementWiseLayer* process3_input = network->addElementWise(*scale3_interpolate->getOutput(0), *process2conv->getOutput(0), ElementWiseOperation::kSUM);
    // process3
    IScaleLayer* process3bn = addBatchNorm2d(network, weightMap, *process3_input->getOutput(0), lname + "process3.0", 1e-5);
    IActivationLayer* process3relu = network->addActivation(*process3bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* process3conv = network->addConvolutionNd(*process3relu->getOutput(0), branch_planes , DimsHW{ 3, 3 }, weightMap[lname + "process3.2.weight"], emptywts);
    assert(process3conv);
    process3conv->setPaddingNd(DimsHW{ 1, 1 });
    //  scale4
    int input_w = input.getDimensions().d[3];
    int input_h = input.getDimensions().d[2];
    IPoolingLayer* scale4pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{ input_h, input_w });
    assert(scale4pool);
    scale4pool->setStrideNd(DimsHW{ input_h, input_w });
    scale4pool->setPaddingNd(DimsHW{ 0, 0 });
    IScaleLayer* scale4bn = addBatchNorm2d(network, weightMap, *scale4pool->getOutput(0), lname + "scale4.1", 1e-5);
    IActivationLayer* scale4relu = network->addActivation(*scale4bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* scale4conv = network->addConvolutionNd(*scale4relu->getOutput(0), branch_planes , DimsHW{ 1, 1 }, weightMap[lname + "scale4.3.weight"], emptywts);
    assert(scale4conv);
    scale4conv->setPaddingNd(DimsHW{ 0, 0 });
    float *deval4 = reinterpret_cast<float*>(malloc(sizeof(float) * branch_planes * input_h * input_w));
    for (int i = 0; i < branch_planes * input_h * input_w; i++) {
        deval4[i] = 1.0;
    }
    Weights deconvwts4{ DataType::kFLOAT, deval4, branch_planes * input_h * input_w };
    IDeconvolutionLayer* scale4_interpolate = network->addDeconvolutionNd(*scale4conv->getOutput(0), branch_planes, DimsHW{ input_h, input_w }, deconvwts4, emptywts);
    scale4_interpolate->setStrideNd(DimsHW{ input_h, input_w });
    scale4_interpolate->setNbGroups(branch_planes);
    IElementWiseLayer* process4_input = network->addElementWise(*scale4_interpolate->getOutput(0), *process3conv->getOutput(0), ElementWiseOperation::kSUM);
    // process4
    IScaleLayer* process4bn = addBatchNorm2d(network, weightMap, *process4_input->getOutput(0), lname + "process4.0", 1e-5);
    IActivationLayer* process4relu = network->addActivation(*process4bn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* process4conv = network->addConvolutionNd(*process4relu->getOutput(0), branch_planes , DimsHW{ 3, 3 }, weightMap[lname + "process4.2.weight"], emptywts);
    assert(process4conv);
    process4conv->setPaddingNd(DimsHW{ 1, 1 });
    //  compression
    ITensor* inputTensors[] = {scale0conv->getOutput(0),  process1conv->getOutput(0) ,  process2conv->getOutput(0), process3conv->getOutput(0), process4conv->getOutput(0)};
    IConcatenationLayer* neck_cat = network->addConcatenation(inputTensors, 5);
    IScaleLayer* compressionbn = addBatchNorm2d(network, weightMap, *neck_cat->getOutput(0), lname + "compression.0", 1e-5);
    IActivationLayer* compressionrelu = network->addActivation(*compressionbn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* compressionconv = network->addConvolutionNd(*compressionrelu->getOutput(0), outplanes , DimsHW{ 1, 1 }, weightMap[lname + "compression.2.weight"], emptywts);
    assert(compressionconv);
    compressionconv->setPaddingNd(DimsHW{ 0, 0 });
    // shortcut
    IScaleLayer* shortcutbn = addBatchNorm2d(network, weightMap, input, lname + "shortcut.0", 1e-5);
    IActivationLayer* shortcutrelu = network->addActivation(*shortcutbn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* shortcutconv = network->addConvolutionNd(*shortcutrelu->getOutput(0), outplanes , DimsHW{ 1, 1 }, weightMap[lname + "shortcut.2.weight"], emptywts);
    assert(shortcutconv);
    shortcutconv->setPaddingNd(DimsHW{ 0, 0 });
    IElementWiseLayer* out = network->addElementWise(*compressionconv->getOutput(0), *shortcutconv->getOutput(0), ElementWiseOperation::kSUM);
    return out;
}

ILayer* segmenthead(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int interplanes, int outplanes, std::string lname)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, input, lname + "bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* conv1 = network->addConvolutionNd(*relu1->getOutput(0), interplanes , DimsHW{ 3, 3 }, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn2", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu2->getOutput(0), outplanes , DimsHW{ 1, 1 }, weightMap[lname + "conv2.weight"], weightMap[lname + "conv2.bias"]);
    assert(conv2);
    conv2->setPaddingNd(DimsHW{ 0, 0 });
    return conv2;
}

}//namespace net
}//namespace camera
}//namespace perception