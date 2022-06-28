#include"yolov4.h"


int main(int argc, char** argv)
{
	std::string model_path = "./yolov4.pt";
	std::string image_path = "street.jpg";
	std::string classes_path = "./coco_classes.txt";
	std::string mode = "image";
	
	std::vector<std::string>  class_names(get_classes(classes_path));
	YOLOV4 yolo(model_path);
	cv::Mat image;
	if (mode=="image")
	{
		image = cv::imread(image_path);
		//boxes = [[ymin,xmin,ymax,xmax,conf,class],[ymin,xmin,ymax,xmax,conf,class]....] 长度为目标数量
		std::vector<std::vector<float>> boxes(yolo.detect_image(image));
		yolo.Show_Detection_Restults(image, boxes, class_names, mode);
		cv::imshow("Object Detection", image);
		if (cv::waitKey(0) == 27) { cv::destroyWindow("Object Detection"); };
	}
	else if (mode=="video")
	{
		cv::VideoCapture cap;
		
		if (image_path=="0")
		{
			cap = cv::VideoCapture(0);
		}
		else
		{
			cap = cv::VideoCapture(image_path);
		}
		if (!cap.isOpened())
		{
			std::cout << "The video opened fail!" << std::endl;
		}
		while (true)
		{
			cap >> image;
			if (image.empty()) break;
			//boxes = [[ymin,xmin,ymax,xmax,conf,class],[ymin,xmin,ymax,xmax,conf,class]....] 长度为目标数量
			std::vector<std::vector<float>> boxes(yolo.detect_image(image));
			yolo.Show_Detection_Restults(image, boxes, class_names, mode);
			cv::imshow("Object Detection", image);
			if (cv::waitKey(30)== 27)break;
		}
		cap.release();
		cv::destroyWindow("Object Detection");
	}
	
	system("pause");
	return 0;

}