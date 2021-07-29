#include <torch/script.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

std::string IMG_PATH = "/home/ibrahim/Projects/Datasets/HPO_Recording/Images/";
std::string IMG_NAME = "1_image_8070.jpg";

std::string MODEL_PATH = "/home/ibrahim/Projects/Gamma_Corrections/Pytorch/SavedModels/";
std::string MODEL_NAME = "test.pt";



int main(int, char**) {
    std::cout << "Hello, world!\n";

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    torch::jit::script::Module module = torch::jit::load(MODEL_PATH + MODEL_NAME);


    cv::Mat img = cv::imread(IMG_PATH + IMG_NAME, 0);
    std::cout << img.size << std::endl;

    cv::Rect roi;
    roi.x = 40;
    roi.y = 20;

    roi.width  = 280;
    roi.height = 250;

    img = img(roi);     
    cv::resize(img, img, cv::Size(100,100));

    cv::namedWindow("frame");
    cv::imshow("frame", img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    std::cout << img.size << std::endl;

    img.convertTo(img, CV_32F, 1.0 / 255.0);




    auto input_tensor = torch::from_blob(img.data, {1, 100, 100, 1}, torch::kFloat32);
    input_tensor = input_tensor.permute({0, 3, 1, 2});

    // input_tensor[0][0] = input_tensor[0][0].div_(450.0);
    // input_tensor[0][1] = input_tensor[0][0].div_(255.0);
    // input_tensor[0][2] = input_tensor[0][0].div_(255.0);
    
    torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();

    std::cout << out_tensor << std::endl;

    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1, 1, 100, 100}));
    // at::Tensor output = module.forward(inputs).toTensor();
    // std::cout << output;


    
    return 0;
}
