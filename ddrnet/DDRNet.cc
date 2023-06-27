//
// Created by qin on 2022/7/14.
//

#include "DDRNet.h"
#include "net_blocks.h"
#include <iostream>
#include <cassert>
#include <cuda_runtime_api.h>
#include <boost/filesystem.hpp>

#define DEVICE 0
#define IN_W 1024
#define IN_H 1024
#define OUT_W 128
#define OUT_H 128
#define IN_BLOB_NAME "input_0"
#define OUT_BLOB_NAME "output_0"

using namespace std;
using namespace perception;
using namespace camera;
using namespace net;

DDRNet::DDRNet(/* args */)
{
    cudaSetDevice(DEVICE);
}

DDRNet::~DDRNet()
{
    delete[] output_;
    delete[] input_;
    engine_->destroy();
}

bool DDRNet::init(Option &option)
{
    option_ = option;
    engine_file_ = option.weight + ".engine." + option.precison;
    LoadEngine();
    input_ = new float[3 * IN_W * IN_H];
    // output_ = new float[4 * OUT_W * OUT_H];
    output_ = new float[4 * IN_W * IN_H];

    return true;
}

map<string, Weights> DDRNet::LoadWeights()
{
    cout << "Loading weights: " << option_.weight << endl;
    map<string, Weights> weightMap;
    // Open weights file
    ifstream input(option_.weight);
    assert(input.is_open() && "Unable to load weight file.");
    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;
        // Read name and type of blob
        string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;
        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

bool DDRNet::LoadEngine()
{
    boost::filesystem::path t_path(engine_file_);
    cout << "Loading: " + engine_file_<< endl;
    if(!boost::filesystem::exists(t_path))
    {
        cout << engine_file_ + " doesn't exist!" << endl;
        return BuildEngine();
    }
    char *trtModelStream = nullptr;
    size_t size = 0;

    ifstream file(engine_file_, ios::binary);
    if (!file.good())
    {
        cerr << "engine: bag engine file!" << endl;
        return false;
    }
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    IRuntime* runtime = createInferRuntime(logger_);
    assert(runtime != nullptr);
    engine_ = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine_ != nullptr);
    // context_ = engine_->createExecutionContext();
    // assert(context_ != nullptr);
    delete[] trtModelStream;

    return true;
}

// Creat the engine using only the API and not any parser.
bool DDRNet::BuildEngine()
{
    unsigned int maxBatchSize = 1;
    DataType dt(DataType::kFLOAT);
    IBuilder* builder = createInferBuilder(logger_);
    IBuilderConfig* config = builder->createBuilderConfig();
    cout<<"----------test-----------"<<endl;
    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(IN_BLOB_NAME, dt, Dims4{ 1, 3, IN_H, IN_W });
    assert(data);
    map<string, Weights> weightMap = LoadWeights();
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 32, DimsHW{ 3, 3 }, weightMap["conv1.0.weight"], weightMap["conv1.0.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ 2, 2 });
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "conv1.1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), 32, DimsHW{ 3, 3 }, weightMap["conv1.3.weight"], weightMap["conv1.3.bias"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ 2, 2 });
    conv2->setPaddingNd(DimsHW{ 1, 1 });
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "conv1.4", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    // layer1
    ILayer* layer1_0 = basicBlock(network, weightMap, *relu2->getOutput(0), 32, 32, 1, false, false, "layer1.0.");
    ILayer* layer1_1 = basicBlock(network, weightMap, *layer1_0->getOutput(0), 32, 32, 1, false, true, "layer1.1.");
    IActivationLayer* layer1_relu = network->addActivation(*layer1_1->getOutput(0), ActivationType::kRELU);
    assert(layer1_relu);
    // layer2
    ILayer* layer2_0 = basicBlock(network, weightMap, *layer1_relu->getOutput(0), 32, 64, 2, true, false, "layer2.0.");
    ILayer* layer2_1 = basicBlock(network, weightMap, *layer2_0->getOutput(0), 64, 64, 1, false, true, "layer2.1."); // 1/8
    IActivationLayer* layer2_relu = network->addActivation(*layer2_1->getOutput(0), ActivationType::kRELU);
    assert(layer2_relu);
    // layer3
    ILayer* layer3_0 = basicBlock(network, weightMap, *layer2_relu->getOutput(0), 64, 128, 2, true, false, "layer3.0.");
    ILayer* layer3_1 = basicBlock(network, weightMap, *layer3_0->getOutput(0), 128, 128, 1, false, true, "layer3.1."); // 1/16
    IActivationLayer* layer3_relu = network->addActivation(*layer3_1->getOutput(0), ActivationType::kRELU);
    assert(layer3_relu);   // layer[2]
    // layer3_
    ILayer* layer3_10 = basicBlock(network, weightMap, *layer2_relu->getOutput(0), 64, 64, 1, false, false, "layer3_.0.");
    ILayer* layer3_11 = basicBlock(network, weightMap, *layer3_10->getOutput(0), 64, 64, 1, false, true, "layer3_.1."); // x_ = self.layer3_(self.relu(layers[1]))
    // down3
    IActivationLayer* down3_input_relu = network->addActivation(*layer3_11->getOutput(0), ActivationType::kRELU);
    assert(down3_input_relu);
    ILayer* down3_out = down3(network, weightMap, *down3_input_relu->getOutput(0), 128, "down3.");
    IElementWiseLayer* down3_add = network->addElementWise(*layer3_1->getOutput(0), *down3_out->getOutput(0), ElementWiseOperation::kSUM);
    ILayer* compression3_input = compression3(network, weightMap, *layer3_relu->getOutput(0), 64, "compression3.");
    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 64 * 2 * 2));
    for (int i = 0; i < 64 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts1{ DataType::kFLOAT, deval, 64 * 2 * 2 };
    IDeconvolutionLayer* compression3_up = network->addDeconvolutionNd(*compression3_input->getOutput(0), 64, DimsHW{ 2, 2 }, deconvwts1, emptywts);
    compression3_up->setStrideNd(DimsHW{ 2, 2 });
    compression3_up->setNbGroups(64);
    IElementWiseLayer* compression3_add = network->addElementWise(*layer3_11->getOutput(0), *compression3_up->getOutput(0), ElementWiseOperation::kSUM);
    // layer4
    IActivationLayer* layer4_input = network->addActivation(*down3_add->getOutput(0), ActivationType::kRELU);
    ILayer* layer4_0 = basicBlock(network, weightMap, *layer4_input->getOutput(0), 128, 256, 2, true, false, "layer4.0.");
    ILayer* layer4_1 = basicBlock(network, weightMap, *layer4_0->getOutput(0), 256, 256, 1, false, true, "layer4.1."); // 1/32
    IActivationLayer* layer4_relu = network->addActivation(*layer4_1->getOutput(0), ActivationType::kRELU);
    assert(layer4_relu);
    // layer4_
    IActivationLayer* layer4_1_input = network->addActivation(*compression3_add->getOutput(0), ActivationType::kRELU);
    ILayer* layer4_10 = basicBlock(network, weightMap, *layer4_1_input->getOutput(0), 64, 64, 1, false, false, "layer4_.0.");
    ILayer* layer4_11 = basicBlock(network, weightMap, *layer4_10->getOutput(0), 64, 64, 1, false, true, "layer4_.1."); // 1/8
    // down4
    IActivationLayer* down4_input_relu = network->addActivation(*layer4_11->getOutput(0), ActivationType::kRELU);
    assert(down4_input_relu);
    ILayer* down4_out = down4(network, weightMap, *down4_input_relu->getOutput(0), 128, "down4.");
    IElementWiseLayer* down4_add = network->addElementWise(*layer4_1->getOutput(0), *down4_out->getOutput(0), ElementWiseOperation::kSUM);
    ILayer* compression4_input = compression4(network, weightMap, *layer4_relu->getOutput(0), 64, "compression4.");
    float *deval2 = reinterpret_cast<float*>(malloc(sizeof(float) * 64 * 4 * 4));
    for (int i = 0; i < 64 * 4 * 4; i++) {
        deval2[i] = 1.0;
    }
    Weights deconvwts2{ DataType::kFLOAT, deval2, 64 * 4 * 4 };
    IDeconvolutionLayer* compression4_up = network->addDeconvolutionNd(*compression4_input->getOutput(0), 64, DimsHW{ 4, 4 }, deconvwts2, emptywts);
    compression4_up->setStrideNd(DimsHW{ 4, 4 });
    compression4_up->setNbGroups(64);
    IElementWiseLayer* compression4_add = network->addElementWise(*layer4_11->getOutput(0), *compression4_up->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* compression4_add_relu = network->addActivation(*compression4_add->getOutput(0), ActivationType::kRELU);
    assert(compression4_add_relu);
    ILayer* layer5_ = Bottleneck(network, weightMap, *compression4_add_relu->getOutput(0), 64, 64, 1, true, true, "layer5_.0.");
    // layer5
    IActivationLayer* layer5_input = network->addActivation(*down4_add->getOutput(0), ActivationType::kRELU);
    assert(layer5_input);
    ILayer* layer5 = Bottleneck(network, weightMap, *layer5_input->getOutput(0), 256, 256, 2, true, true, "layer5.0.");
    ILayer* ssp = DAPPM(network, weightMap, *layer5->getOutput(0), 512, 128, 128, "spp.");
    float *deval3 = reinterpret_cast<float*>(malloc(sizeof(float) * 128 * 8 * 8));
    for (int i = 0; i < 128 * 8 * 8; i++) {
        deval3[i] = 1.0;
    }
    Weights deconvwts3{ DataType::kFLOAT, deval3, 128 * 8 * 8 };
    IDeconvolutionLayer* spp_up = network->addDeconvolutionNd(*ssp->getOutput(0), 128, DimsHW{ 8, 8 }, deconvwts3, emptywts);
    spp_up->setStrideNd(DimsHW{ 8, 8 });
    spp_up->setNbGroups(128);
    IElementWiseLayer* final_in = network->addElementWise(*spp_up->getOutput(0), *layer5_->getOutput(0), ElementWiseOperation::kSUM);
    //gai
    ILayer* seg_out = segmenthead(network, weightMap, *final_in->getOutput(0), 64, 4, "final_layer.");

    IResizeLayer* upsample = network->addResize(*seg_out->getOutput(0));
    upsample->setOutputDimensions(Dims4{1, 4, 1024, 1024});
    upsample->setResizeMode(ResizeMode::kLINEAR);
    upsample->setAlignCorners(true);
    upsample->getOutput(0)->setName(OUT_BLOB_NAME);
    network->markOutput(*upsample->getOutput(0));
    //==================================================================================================================
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB

    if(option_.precison == "fp16")
        config->setFlag(BuilderFlag::kFP16);

    cout << "Building engine, please wait for a while..." << endl;
    engine_ = builder->buildEngineWithConfig(*network, *config);
    assert(engine_ != nullptr);
    cout << "Build engine successfully!" << endl;
    ofstream save_engine(engine_file_, ios::binary);
    if (!save_engine)
        cerr << "could not open plan output file" << endl;
    IHostMemory* modelStream = engine_->serialize();
    save_engine.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    network->destroy();
    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    builder->destroy();

    return true;
}

void DDRNet::cvimage2input(cv::Mat &image)
{
    cv::Mat input_image;
    image.copyTo(input_image);
    cv::resize(input_image, input_image, cv::Size(IN_W, IN_H));
    for (int row = 0, i = 0; row < IN_H; row++)
    {
        uchar* uc_pixel = input_image.data + row * input_image.step;
        for (int col = 0; col < IN_W; col++, i++)
        {
            static vector<float> std_value = option_.std_value;
            static vector<float> mean_value = option_.mean_value;
            for(int c = 0; c < 3; c++)
                input_[i + c * IN_W * IN_H] = (uc_pixel[2-c]/255.f - mean_value[2-c]) / std_value[2-c];
            uc_pixel += 3;
        }
    }
}

cv::Mat DDRNet::segment(cv::Mat &image)
{
    cvimage2input(image);
    // const ICudaEngine& engine = context_->getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine_->getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine_->getBindingIndex(IN_BLOB_NAME);
    const int outputIndex = engine_->getBindingIndex(OUT_BLOB_NAME);

    // Create GPU buffers on device
    static int input_size = 3*IN_W*IN_H*sizeof(float);
    // static int output_size = 4*OUT_W*OUT_H*sizeof(float);
    static int output_size = 4*IN_W*IN_H*sizeof(float);
    CHECK(cudaMalloc(&buffers[inputIndex], input_size));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input_, input_size, cudaMemcpyHostToDevice, stream));
    static IExecutionContext* context = engine_->createExecutionContext();
    // context->getEngine();
    context->enqueue(1, buffers, stream, nullptr);
    //gai
    CHECK(cudaMemcpyAsync(output_, buffers[outputIndex], output_size, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    cv::Mat result = ParseOutput();
    cv::resize(result, result, image.size(), 0, 0, cv::INTER_NEAREST);
    return result;
}

// cv::Mat DDRNet::ParseOutput()
// {
//     cv::Mat mask(OUT_H, OUT_W, CV_32FC4);
//     for (int i = 0; i < OUT_H; ++i)
//     {
//         // cv::Vec<float, 4> *mask_ptr = mask.ptr<cv::Vec<float, 4>>(i);
//         for (int j = 0; j < OUT_W; ++j)
//         {
//             for (int c = 0; c < 4; ++c)
//             {
//                 mask.at<cv::Vec4f>(j, i)[c] = output_[c * OUT_H * OUT_W + i * OUT_W + j];
//             }
//         }
//     }
//     cv::resize(mask, mask, cv::Size(IN_H, IN_W));
//
//     cv::Mat index_image(IN_H, IN_W, CV_8UC1, cv::Scalar(0));
//     for (int i = 0; i < IN_H; ++i)
//     {
//         cv::Vec<uchar, 1> *row_ptr = index_image.ptr<cv::Vec<uchar, 1>>(i);
//         for (int j = 0; j < IN_W; ++j)
//         {
//             float max_value = mask.at<cv::Vec4f>(j, i)[0];
//             for (int c = 1; c < 4; ++c)
//             {
//                 float value = mask.at<cv::Vec4f>(j, i)[c];
//                 if(value < max_value)
//                     continue;
//                 max_value = value;
//                 row_ptr[j] = c;
//             }
//         }
//     }
//     return index_image;
// }

cv::Mat DDRNet::ParseOutput()
{
    cv::Mat mask(IN_H, IN_W, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < IN_H; ++i)
    {
        cv::Vec<uchar, 1> *row_ptr = mask.ptr<cv::Vec<uchar, 1>>(i);
        for (int j = 0; j < IN_W; ++j)
        {
            float max_value = output_[i * IN_W + j];
            for (int c = 1; c < 4; ++c)
            {
                float value = output_[c * IN_W * IN_H + i * IN_W + j];
                if(max_value > value)
                    continue;
                max_value = value;
                row_ptr[j] = c;
            }
        }
    }

    return mask;
}

cv::Vec3b MAP[4] = {cv::Vec3b(0,0,0), cv::Vec3b(0,0,255), cv::Vec3b(0,255,255), cv::Vec3b(255,0,0)};

cv::Mat DDRNet::ColorMap(cv::Mat &src)
{
    cv::Mat dst(src.size(), CV_8UC3);
    for(int i = 0; i < src.cols; i++)
    {
        for(int j = 0; j < src.rows; j++)
            dst.at<cv::Vec3b>(j, i) = MAP[src.at<uchar>(j, i)];
    }
    return dst;
}
