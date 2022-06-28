#pragma once
#include<stdio.h>
#include<torch/torch.h>
#include<torch/script.h>
#include<string>
#include<iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include<vector>
#include"utils.h"



bool is_cuda_available();
std::vector<std::string> get_classes(std::string &path);

class YOLOV4
{
public:
	YOLOV4(std::string&model_path);
	std::vector<std::vector<float>> detect_image(cv::Mat image);
	void Show_Detection_Restults(cv::Mat image,std::vector<std::vector<float>>boxes,std::vector<std::string>class_names,std::string mode);
private:
	torch::jit::script::Module model;
	float all_anchors[3][3][2] = { {{142, 110}, {192, 243}, {459, 401}},
								  {{36,   75}, {76,   55}, {72,  146}},
								  {{12,   16}, {19,   36}, {40,   28}} };
	float model_image_size[2] = { INPUT_SHAPE, INPUT_SHAPE };
};
