
#define NUM_CLASSES 80
#define CONF_THRES 0.6
#define NMS_THRES 0.5
#define INPUT_SHAPE 416

at::Tensor yolo_correct_boxes(at::Tensor top, at::Tensor left, at::Tensor bottom, at::Tensor right, at::Tensor model_image_size, at::Tensor src_image_size);
class DecodeBox
{
public:
	DecodeBox(float t_anchors[][2], float t_image_size[]);
	at::Tensor decode_box(at::Tensor input);
private:
	/**************************************************************************
	 * 存放每个特征层的先验框
	 **************************************************************************/
	float anchors[3][2];

	float image_size[2];

	int num_anchors = 3;
	
	int num_classes = NUM_CLASSES; //替换为自己的类
	
	int bbox_attrs = num_classes + 5;
};

cv::Mat letterbox_image(cv::Mat image, float size[]);

at::Tensor nms_cpu(const at::Tensor& dets,const at::Tensor& scores,const float threshold);

std::vector<at::Tensor> yolo_nms(at::Tensor prediction, int num_classes, float conf_thres = 0.5, float nms_thres = 0.4);

