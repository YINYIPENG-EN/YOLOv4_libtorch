#include"yolov4.h"

template <typename scalar_t>
at::Tensor nms_cpu_kernel(const at::Tensor& dets,
	const at::Tensor& scores,
	const float threshold) 
{
	/*AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
	AT_ASSERTM(!scores.scalar_type().is_cuda(), "scores must be a CPU tensor");
	AT_ASSERTM(dets.scarlar_type() == scores.scarlar_type(), "dets should have the same type as scores");
*/
	if (dets.numel() == 0) {
		return at::empty({ 0 }, dets.options().dtype(at::kLong).device(at::kCPU));
	}

	auto x1_t = dets.select(1, 0).contiguous();
	auto y1_t = dets.select(1, 1).contiguous();
	auto x2_t = dets.select(1, 2).contiguous();
	auto y2_t = dets.select(1, 3).contiguous();

	at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

	auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

	auto ndets = dets.size(0);
	at::Tensor suppressed_t = at::zeros({ ndets }, dets.options().dtype(at::kByte).device(at::kCPU));

	auto suppressed = suppressed_t.data_ptr<uint8_t>();
	auto order = order_t.data_ptr<int64_t>();
	auto x1 = x1_t.data_ptr<scalar_t>();
	auto y1 = y1_t.data_ptr<scalar_t>();
	auto x2 = x2_t.data_ptr<scalar_t>();
	auto y2 = y2_t.data_ptr<scalar_t>();
	auto areas = areas_t.data_ptr<scalar_t>();

	for (int64_t _i = 0; _i < ndets; _i++) {
		auto i = order[_i];
		if (suppressed[i] == 1)
			continue;
		auto ix1 = x1[i];
		auto iy1 = y1[i];
		auto ix2 = x2[i];
		auto iy2 = y2[i];
		auto iarea = areas[i];

		for (int64_t _j = _i + 1; _j < ndets; _j++) {
			auto j = order[_j];
			if (suppressed[j] == 1)
				continue;
			auto xx1 = std::max(ix1, x1[j]);
			auto yy1 = std::max(iy1, y1[j]);
			auto xx2 = std::min(ix2, x2[j]);
			auto yy2 = std::min(iy2, y2[j]);

			auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + 1);
			auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + 1);
			auto inter = w * h;
			auto ovr = inter / (iarea + areas[j] - inter);
			if (ovr >= threshold)
				suppressed[j] = 1;
		}
	}
	return at::nonzero(suppressed_t == 0).squeeze(1);
}

at::Tensor nms_cpu(const at::Tensor& dets,
	const at::Tensor& scores,
	const float threshold) {
	at::Tensor result;
	AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms", [&] {
		result = nms_cpu_kernel<scalar_t>(dets, scores, threshold);
	});
	return result;
}

DecodeBox::DecodeBox(float t_layer_anchors[][2], float t_model_image_size[])
{
	// ��ȡ��ǰ����ͼ��anchor����
	for (size_t i = 0; i < 3; i++)
		for (size_t j = 0; j < 2; j++)
			anchors[i][j] = t_layer_anchors[i][j];

	// ��ȡģ����ͼ��ĳߴ�
	for (size_t i = 0; i < 2; i++)
		image_size[i] = t_model_image_size[i];
}
at::Tensor DecodeBox::decode_box(at::Tensor input)
{
	// ��ȡ�ߴ�Ȳ���
	int batch_size = input.size(0);
	int input_height = input.size(2);
	int input_width = input.size(3);
	// ����
	int stride_w = image_size[0] / input_width;
	int stride_h = image_size[1] / input_height;

	// ��ʱ��õ�scaled_anchors��С�������ÿ���������
	float scaled_anchors[3][2];
	for (int i = 0; i < 3; i++)
	{
		scaled_anchors[i][0] = anchors[i][0] / stride_w;
		scaled_anchors[i][1] = anchors[i][1] / stride_h;
	}

	// (bs, 3*(5+num_classes), h, w)  -->  (bs, 3, h, w, (5+num_classes))
	at::Tensor prediction = input.view({ batch_size, num_anchors, bbox_attrs, input_height, input_width }).permute({ 0, 1, 3, 4, 2 }).contiguous();
	// ���������λ�õĵ�������
	at::Tensor x = at::sigmoid(prediction.index({ "...", 0 }));
	at::Tensor y = at::sigmoid(prediction.index({ "...", 1 }));
	// ������߲�������
	at::Tensor w = prediction.index({ "...", 2 });
	at::Tensor h = prediction.index({ "...", 3 });
	// ���ŶȻ�ȡ
	at::Tensor conf = at::sigmoid(prediction.index({ "...", 4 }));
	// ����������Ŷ�
	at::Tensor pred_cls = at::sigmoid(prediction.index({ "...",torch::indexing::Slice{5, torch::indexing::None} }));

	// ��������  ��������ģ��������Ͻ� bs, 3, h, w
	at::Tensor grid_x = at::linspace(0, input_width - 1, input_width).repeat({ input_width, 1 }).repeat({ batch_size*num_anchors, 1, 1 }).view({ x.sizes() }).toType(torch::kFloat);
	at::Tensor grid_y = at::linspace(0, input_height - 1, input_height).repeat({ input_height, 1 }).t().repeat({ batch_size*num_anchors, 1, 1 }).view({ y.sizes() }).toType(torch::kFloat);

	// ���������ʽ���������Ŀ��  ����ת��Ϊtensor   ����shape  bs, 3, h, w
	at::Tensor anchor_w = at::from_blob(scaled_anchors, { 3, 2 }, at::kFloat).index_select(1, at::tensor(0).toType(at::kLong))\
		.repeat({ batch_size, input_height*input_width }).view(w.sizes());
	at::Tensor anchor_h = at::from_blob(scaled_anchors, { 3, 2 }, at::kFloat).index_select(1, at::tensor(1).toType(at::kLong))\
		.repeat({ batch_size, input_height*input_width }).view(h.sizes());
	/*
	����Ԥ�������������е���
    ���ȵ������������ģ�����������������½�ƫ���ٵ��������Ŀ�ߡ�
	*/
	at::Tensor pred_boxes = at::zeros({ prediction.index({"...", torch::indexing::Slice({torch::indexing::None, 4})}).sizes() }).toType(at::kFloat);
	// ������������ͼ�ϵĳߴ�ֵ
	pred_boxes.index_put_({ "...", 0 }, (x.data() + grid_x));
	pred_boxes.index_put_({ "...", 1 }, (y.data() + grid_y));
	pred_boxes.index_put_({ "...", 2 }, (at::exp(w.data()) * anchor_w));
	pred_boxes.index_put_({ "...", 3 }, (at::exp(h.data()) * anchor_h));

	// ����ת��tensor  (batch_size, 6) -->  (batch_size, (x, y, w, h, conf, pred_cls))
	at::Tensor _scale = at::tensor({ stride_w, stride_h, stride_w, stride_h }).toType(at::kFloat);
	//����������ƴ��
	at::Tensor output = at::cat({ pred_boxes.view({batch_size, -1, 4}) * _scale, \
								conf.view({batch_size, -1, 1}), \
								pred_cls.view({batch_size, -1, num_classes}) }, - 1);

	return output.data();
};

cv::Mat letterbox_image(cv::Mat image, float size[])
{
	// ͼƬ��ʵ��С
	float iw = image.cols, ih = image.rows;
	// ��������ͼƬ�Ĵ�С
	float w = size[0], h = size[1];
	float scale = std::min(w / iw, h / ih);
	// ������Ĵ�С
	int nw = int(iw * scale), nh = int(ih * scale);
	// 
	cv::resize(image, image, { nw, nh });
	// ����ͼƬ
	cv::Mat new_image(w, h, CV_8UC3, cv::Scalar(128, 128, 128));
	// ���û����������򲢸���
	cv::Rect roi_rect = cv::Rect((w - nw) / 2, (h - nh) / 2, nw, nh);
	image.copyTo(new_image(roi_rect));
	return new_image;
};

at::Tensor yolo_correct_boxes(at::Tensor top, at::Tensor left, at::Tensor bottom, at::Tensor right, at::Tensor model_image_size, at::Tensor src_image_size)
{
	// ����������ź�ĳߴ磬�������Ҷ���  shape (1, 2) (h ,w)
	at::Tensor new_shape = (src_image_size * at::amin(model_image_size / src_image_size, 0));
	// ���㳤���Ӧ�����ű���
	at::Tensor scale = model_image_size / new_shape;

	// �����ڻҶ����ϵ�ƫ��
	at::Tensor offset = (model_image_size - new_shape) / 2. / model_image_size;    // ��һ����

	// �γ����ĵ㣨y��x������ shape = n,2
	at::Tensor box_yx = at::cat({ (top + bottom) / 2, (left + right) / 2 }, -1) / model_image_size;

	// �γɿ�߲��� shape = n,2
	at::Tensor box_hw = at::cat({ bottom - top, right - left }, -1) / model_image_size;

	// ����ָ���������ʵͼƬ������ĵ����꼰����
	box_yx = (box_yx - offset) * scale;
	box_hw = box_hw * scale;

	// 
	at::Tensor box_mins = (box_yx - box_hw / 2.) * src_image_size;
	at::Tensor box_maxs = (box_yx + box_hw / 2.) * src_image_size;

	// ymin ximn ymax xmax
	at::Tensor boxes = at::cat({ box_mins, box_maxs }, -1);

	return boxes;
};



std::vector<at::Tensor> yolo_nms(at::Tensor prediction, int num_classes, float conf_thres, float nms_thres)
{
	//(bs, 3*(13*13+26*26+52*52)�� 5+num_classes)
	at::Tensor box_corner = at::zeros(prediction.sizes());
	// �����ϽǺ����½�
	box_corner.index_put_({ "...", 0 }, prediction.index({ "...", 0 }) - prediction.index({ "...", 2 }) / 2);
	box_corner.index_put_({ "...", 1 }, prediction.index({ "...", 1 }) - prediction.index({ "...", 3 }) / 2);
	box_corner.index_put_({ "...", 2 }, prediction.index({ "...", 0 }) + prediction.index({ "...", 2 }) / 2);
	box_corner.index_put_({ "...", 3 }, prediction.index({ "...", 1 }) + prediction.index({ "...", 3 }) / 2);
	// ��ֵ x1 y1 x2 y2
	prediction.index_put_({ "...", torch::indexing::Slice(torch::indexing::None,4) }, box_corner.index({ "...", torch::indexing::Slice(torch::indexing::None,4) }));
	
	std::vector<at::Tensor> nms_output;
	at::Tensor output = prediction[0];
	std::tuple<at::Tensor, at::Tensor> temp = at::max(output.index({ "...", torch::indexing::Slice(5, 5 + num_classes) }), 1, true);
	at::Tensor class_conf = std::get<0>(temp);
	at::Tensor class_pred = std::get<1>(temp);

	// �������Ŷ�ɸѡ
	at::Tensor conf_mask = (output.index({ "...", 4 }) * class_conf.index({ "...", 0 }) >= conf_thres).squeeze();

	// ������Ŀ��Ĳ���
	output = output.index({ conf_mask });

	// û��Ŀ�ֱ꣬�ӷ��ؿս��
	if (output.size(0) == 0)
	{
		return nms_output;
	}
	class_conf = class_conf.index({ conf_mask });
	class_pred = class_pred.index({ conf_mask });
	// ��õ�����Ϊ(x1, y1, x2, y2, obj_conf, class_conf, class_pred)
	at::Tensor detections = at::cat({ output.index({"...", torch::indexing::Slice(torch::indexing::None, 5)}), class_conf.toType(at::kFloat), class_pred.toType(at::kFloat) }, -1);
	std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_labels_tuple = at::unique_consecutive(detections.index({ "...", -1 }));
	at::Tensor unique_labels = std::get<0>(unique_labels_tuple);
	// �������е�����
	for (int i = 0; i < unique_labels.size(0); i++)
	{
		// ��ȡĳ�������ɸѡ���Ԥ����
		at::Tensor detections_class = detections.index({ detections.index({"...", -1}) == unique_labels[i] });
		at::Tensor keep = nms_cpu(detections_class.index({ "...", torch::indexing::Slice(torch::indexing::None,4) }), detections_class.index({ "...", 4 })*detections_class.index({ "...", 5 }), nms_thres);
		at::Tensor max_detection = detections_class.index({ keep });
		if (i == 0)
		{
			nms_output.push_back(max_detection);
		}
		else
		{
			nms_output[0] = at::cat({ nms_output[0], max_detection });
		}

	}
	return nms_output;
}
