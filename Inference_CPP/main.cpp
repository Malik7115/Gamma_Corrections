#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include </home/ibrahim/Softwares/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h>
#include </home/ibrahim/Softwares/onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_api.h>


using namespace cv;
using namespace dnn;

std::string IMG_PATH = "/home/ibrahim/Projects/Datasets/HPO_Recording/Images/";
std::string IMG_NAME = "1_image_15.jpg";

std::string MODEL_PATH = "/home/ibrahim/Projects/Gamma_Corrections/Pytorch/SavedModels/";
std::string MODEL_NAME = "gamma_correction.onnx";

int main() {

    Ort::Env env;
    Ort::Session session{ env, "gamma_correction.onnx", Ort::SessionOptions{} };
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    const char* inputName = session.GetInputName(0, allocator);
    std::cout << "Input Name: " << inputName << std::endl;

    



    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    std::cout << "Hello, world! Touchdown\n\n";
    Mat image, blob;
    // Net net = readNetFromONNX(MODEL_NAME);

    image = imread(IMG_PATH + IMG_NAME, 1);


    imshow("Display Image", image);
    waitKey(0);
    destroyAllWindows();

    resize(image, image, Size(100,100));
    cv::dnn::blobFromImage(image, blob);


    std::cout << image.size << std::endl;
    std::cout << blob.size << std::endl;

    // net.setInput(blob);
    // Mat prob = net.forward();


    return 0;

}
