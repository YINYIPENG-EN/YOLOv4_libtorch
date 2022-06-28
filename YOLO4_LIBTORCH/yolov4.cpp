#include"yolov4.h"

int fontFace = cv::FONT_HERSHEY_SIMPLEX;  // 字体
double fontScale = 0.6;   // 字体大小
int thickness = 2;  // 粗细
int baseline;

bool is_cuda_available()
{
	if (torch::cuda::is_available())
	{
		std::cout << "The cuda is available!" << std::endl;
		return true;
	}
	else
	{
		std::cout << "The cuda isn't available!" << std::endl;
		return false;
	}
};

YOLOV4::YOLOV4(std::string& model_path)
{
	
	model = torch::jit::load(model_path);
	if (is_cuda_available())
	{
		model.to(torch::kCUDA);
	}
	else
	{
		model.to(torch::kCPU);
	}
}
std::vector<std::vector<float>> YOLOV4::detect_image(cv::Mat image)
{
	DecodeBox yolo_decodes1(all_anchors[0], model_image_size);
	DecodeBox yolo_decodes2(all_anchors[1], model_image_size);
	DecodeBox yolo_decodes3(all_anchors[2], model_image_size);
	// 调整图片大小
	cv::Mat crop_img = letterbox_image(image, model_image_size);
	// 调整图片格式
	cv::cvtColor(crop_img, crop_img, cv::COLOR_BGR2RGB);
	crop_img.convertTo(crop_img, CV_32FC3, 1.f / 255.f);
	// 转换为tensor
	auto tensor_image = at::from_blob(crop_img.data, { 1, crop_img.rows, crop_img.cols, 3 }).to(torch::kCUDA);
	tensor_image = tensor_image.permute({ 0,3,1,2 }).contiguous();
	// 输入初始化
	std::vector<torch::jit::IValue> input;
	input.emplace_back(tensor_image);

	auto outputs = model.forward(input).toTuple();
	// 提取三个head
	std::vector<at::Tensor> output(3);
	output[0] = outputs->elements()[0].toTensor().to(at::kCPU);
	output[1] = outputs->elements()[1].toTensor().to(at::kCPU);
	output[2] = outputs->elements()[2].toTensor().to(at::kCPU);

	std::vector<at::Tensor> feature_out(3);
	for (size_t i = 0; i < 3; i++)
	{
		if (i == 0) feature_out[0] = yolo_decodes1.decode_box(output[0]);
		if (i == 1) feature_out[1] = yolo_decodes2.decode_box(output[1]);
		if (i == 2) feature_out[2] = yolo_decodes3.decode_box(output[2]);
	}

	// 在第二维度上做拼接， shape： (bs, 3*(13*13+26*26+52*52)， 5+num_classes)
	at::Tensor out = at::cat({ feature_out[0], feature_out[1], feature_out[2] }, 1);
	// 得到nms的输出  （输出，类别数，置信度阈值，nms-iou阈值）
	std::vector<at::Tensor> nms_out = yolo_nms(out, NUM_CLASSES, CONF_THRES, NMS_THRES);
	
	// 没有目标，返回为空
	if (nms_out.size() == 0)
	{
		std::vector<std::vector<float>> temp;
		return temp;
	}

	// 内容为(x1, y1, x2, y2, obj_conf, class_conf, class_pred)
	at::Tensor bbox = nms_out[0];

	// 提取置信度及类别
	at::Tensor top_conf = (bbox.index({ "...", 4 }) * bbox.index({ "...", 5 })).unsqueeze(-1);  // 提取置信度
	at::Tensor top_label = bbox.index({ "...", -1 }).unsqueeze(-1);  // 提取所属类别
	// 分别提取定点坐标，并拓展维度
	// at::Tensor top_bbox = bbox.index({"...", Slice(None, 4)});  // 提取坐标
	at::Tensor box_xmin = bbox.index({ "...", 0 }).unsqueeze(-1);
	at::Tensor box_ymin = bbox.index({ "...", 1 }).unsqueeze(-1);
	at::Tensor box_xmax = bbox.index({ "...", 2 }).unsqueeze(-1);
	at::Tensor box_ymax = bbox.index({ "...", 3 }).unsqueeze(-1);

	// 提取原始图片的宽高
	std::vector<int> image_shape(2, 0);
	image_shape[0] = image.rows;  // h
	image_shape[1] = image.cols;  // w
	// 恢复到原始的尺寸上
	at::Tensor bboxes = yolo_correct_boxes(box_ymin, box_xmin, box_ymax, box_xmax, \
		at::from_blob(model_image_size, { 2 }, at::kFloat), \
		at::from_blob(image_shape.data(), { 2 }, at::kInt).toType(at::kFloat));
	// // 与置信度和类别整合  n * (ymin, xmin, ymax, xmax, conf, class)
	bboxes = at::cat({ bboxes, top_conf, top_label }, -1);

	// tensor转换为数组 (n, 6)  
	std::vector<std::vector<float>> boxes(bboxes.sizes()[0], std::vector<float>(6));

	for (int i = 0; i < bboxes.sizes()[0]; i++)
	{

		for (int j = 0; j < 6; j++)
		{
			boxes[i][j] = bboxes.index({ at::tensor(i).toType(at::kLong),at::tensor(j).toType(at::kLong) }).item().toFloat();
			// cout << boxes[i][j] << endl;
		}

	}
	return boxes;
};

std::vector<std::string> get_classes(std::string &path)
{
	std::vector<std::string> classes_name;
	std::ifstream infile(path);
	if (infile.is_open())
	{
		std::string line;
		while (getline(infile, line))
		{
			classes_name.emplace_back(line);
		}
	}
	return classes_name;
}

void YOLOV4::Show_Detection_Restults(cv::Mat image,std::vector < std::vector<float>>boxes, std::vector<std::string>class_names,std::string mode)
{
	
	for (size_t i = 0; i < boxes.size(); i++)
	{
		// 打印种类及位置信息
		std::cout << class_names[int(boxes[i][5])] << ": (xmin:" \
			<< boxes[i][1] << ", ymin:" << boxes[i][0] << ", xmax:" << boxes[i][3] << ", ymax:" << boxes[i][2] << ") --" \
			<< "confidence: " << boxes[i][4] << std::endl;
		// 计算位置
		cv::Rect rect(int(boxes[i][1]), int(boxes[i][0]), int(boxes[i][3] - boxes[i][1]), int(boxes[i][2] - boxes[i][0]));
		cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 1, cv::LINE_8, 0);
		// 获取文本框的大小
		cv::Size text_size = cv::getTextSize(class_names[int(boxes[i][5])], fontFace, fontScale, thickness, &baseline);
		// 绘制的起点
		cv::Point origin;
		origin.x = int(boxes[i][1]);
		origin.y = int(boxes[i][0]) + text_size.height;
		// cv::putText(InputOutputArray img, const String &text, Point org, int fontFace, double fontScale, Scalar color)
		cv::putText(image, class_names[int(boxes[i][5])], origin, fontFace, fontScale, cv::Scalar(0, 0, 255), thickness);

		// 置信度显示
		std::string text = std::to_string(boxes[i][4]);
		text = text.substr(0, 5);
		cv::Size text_size2 = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
		origin.x = origin.x + text_size.width + 3;
		origin.y = int(boxes[i][0]) + text_size2.height;
		cv::putText(image, text, origin, fontFace, fontScale, cv::Scalar(0, 0, 255), thickness);
	}
	// 如果没检测到任何目标
	if (boxes.size() == 0)
	{
		const std::string text = "NO object";
		float fontScale = 2.0;
		// 获取文本框的大小
		cv::Size text_size = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
		std::cout << "no target detected!" << std::endl;
		cv::Point origin; // 绘制的起点
		origin.x = 0;
		origin.y = 0 + text_size.height;
		// cv::putText(InputOutputArray img, const String &text, Point org, int fontFace, double fontScale, Scalar color)
		cv::putText(image, text, origin, fontFace, fontScale, cv::Scalar(255, 0, 0), thickness);
	}

};

