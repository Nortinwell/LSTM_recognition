#include <caffe\caffe.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

/*
layers register
*/
#include "caffe/common.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"

#include "caffe/layers/permute_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"

#include "caffe/layers/concat_layer.hpp"

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe\layers\lstm_layer.hpp"
#include "caffe\layers\continuation_indicator_layer.hpp"
#include "caffe\layers\permute_layer.hpp"
#include "caffe\layers\slice_layer.hpp"
#include "caffe\layers\scale_layer.hpp"
#include "caffe\layers\eltwise_layer.hpp"
#include "caffe\layers\recurrent_layer.hpp"
#include "caffe\layers\bias_layer.hpp"
#include "caffe\layers\parameter_layer.hpp"
#include "caffe\layers\split_layer.hpp"
#include "caffe\layers\crop_layer.hpp"
#include "caffe\layers\concat_layer.hpp"
#include "caffe\util\format.hpp"
#include "caffe\layers\rnn_layer.hpp"
#include "caffe\layers\reduction_layer.hpp"
#include "caffe\layers\reverse_layer.hpp"

namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(ConvolutionLayer);
	REGISTER_LAYER_CLASS(Convolution);
	extern INSTANTIATE_CLASS(ReLULayer);
	REGISTER_LAYER_CLASS(ReLU);
	extern INSTANTIATE_CLASS(PoolingLayer);
	REGISTER_LAYER_CLASS(Pooling);
	extern INSTANTIATE_CLASS(LRNLayer);
	REGISTER_LAYER_CLASS(LRN);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	REGISTER_LAYER_CLASS(Softmax);

	//REGISTER_LAYER_CLASS(Normalize);
	extern INSTANTIATE_CLASS(PermuteLayer);
	//REGISTER_LAYER_CLASS(Permute);
	extern INSTANTIATE_CLASS(FlattenLayer);
	//REGISTER_LAYER_CLASS(Flatten);

	//REGISTER_LAYER_CLASS(PriorBox);
	extern INSTANTIATE_CLASS(ReshapeLayer);
	//REGISTER_LAYER_CLASS(Reshape);
	extern INSTANTIATE_CLASS(ConcatLayer);
	//REGISTER_LAYER_CLASS(Concat);

	//REGISTER_LAYER_CLASS(DetectionOutput);
	extern INSTANTIATE_CLASS(BatchNormLayer);
	extern INSTANTIATE_CLASS(DeconvolutionLayer);
	extern INSTANTIATE_CLASS(ContinuationIndicatorLayer);
	extern INSTANTIATE_CLASS(EltwiseLayer);
	extern INSTANTIATE_CLASS(LSTMLayer);
	extern INSTANTIATE_CLASS(LSTMUnitLayer);
	extern INSTANTIATE_CLASS(SliceLayer);
	extern INSTANTIATE_CLASS(ScaleLayer);
	extern INSTANTIATE_CLASS(RecurrentLayer);
	extern INSTANTIATE_CLASS(BiasLayer);
	extern INSTANTIATE_CLASS(ParameterLayer);
	extern INSTANTIATE_CLASS(SplitLayer);
	extern INSTANTIATE_CLASS(CropLayer);
	extern INSTANTIATE_CLASS(ConcatLayer);
	extern INSTANTIATE_CLASS(RNNLayer);
	extern INSTANTIATE_CLASS(ReductionLayer);
	extern INSTANTIATE_CLASS(ReverseLayer);
	//REGISTER_LAYER_CLASS(Deconvolution);
	//(Python,Creator_PythonLayer<float>);

}
using namespace std;
using namespace cv;
using namespace caffe;

const int BLANK_LABEL = 47;

string labelChar[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D",
"E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
"a", "k", "g", "!", "+", "/", "(", ")", "*", "[","]" };

class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
		int blank_label);

	std::vector<int> Classify(const cv::Mat& img);

private:
	std::vector<int> Predict(const cv::Mat& img);
	void GetLabelseqs(const std::vector<int>& label_seq_with_blank,
		std::vector<int>& label_seq);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	int blank_label_;
	boost::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
};

Classifier::Classifier(const string& model_file,
	const string& trained_file,
	int blank_label) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif
	blank_label_ = blank_label;
	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

void Classifier::GetLabelseqs(const std::vector<int>& label_seq_with_blank,
	std::vector<int>& label_seq) {
	label_seq.clear();
	int prev = blank_label_;
	int length = label_seq_with_blank.size();
	for (int i = 0; i < length; ++i) {
		int cur = label_seq_with_blank[i];
		if (cur != prev && cur != blank_label_) {
			label_seq.push_back(cur);
		}
		prev = cur;
	}
}

/* Return the top N predictions. */
std::vector<int> Classifier::Classify(const cv::Mat& img) {
	std::vector<int> pred_label_seq_with_blank = Predict(img);
	std::vector<int> pred_label_seq;
	GetLabelseqs(pred_label_seq_with_blank, pred_label_seq);
	return pred_label_seq;
}

std::vector<int> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
//	input_geometry_.height = img.rows;
//	input_geometry_.width = img.cols;

//	cout << "----------------height:" << input_geometry_.height << " ======width:" << input_geometry_.width << endl;
	input_layer->Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);

	//boost::shared_ptr<Layer<float>> indicator = net_->layer_by_name("indicator");
	//caffe::ContinuationIndicatorParameter c(indicator->layer_param().continuation_indicator_param());
	//c.set_time_step(29);
	
	//indicator->layer_param().continuation_indicator_param().set_time_step(27);
	
	//const int time_step2 = c.time_step();
	
	//cout << "time:::::----------------" << time_step << " " << time_step2 << endl;



	//input_layer->Reshape(1, num_channels_,img.rows, img.cols);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];

	//const int time_step = input_geometry_.width;

	 
	//indicator->layer_param().continuation_indicator_param();
	
//	indicator->layer_param().continuation_indicator_param().set_time_step(40);
	boost::shared_ptr<Layer<float>> indicator = net_->layer_by_name("indicator");
	const int time_step = indicator->layer_param().continuation_indicator_param().time_step();
	//const int time_step = input_geometry_.width/2;
	const int alphabet_size = output_layer->shape(2);


	std::vector<int> pred_label_seq_with_blank(time_step);
	const float* pred_data = output_layer->cpu_data();

	for (int t = 0; t < time_step; ++t) {
		pred_label_seq_with_blank[t] = std::max_element(pred_data, pred_data + alphabet_size) - pred_data;
		pred_data += alphabet_size;
	}

	return pred_label_seq_with_blank;
}


void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	sample_float /= 255.0;

	cv::split(sample_float, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}
string dir_path = "";
vector<string> fileNames;

//×Ö·û´®·Ö¸îº¯Êý
vector< string> split(string str, string pattern)
{
	vector<string> ret;
	if (pattern.empty()) return ret;
	size_t start = 0, index = str.find_first_of(pattern, 0);
	while (index != str.npos)
	{
		if (start != index)
			ret.push_back(str.substr(start, index - start));
		start = index + 1;
		index = str.find_first_of(pattern, start);
	}
	if (!str.substr(start).empty())
		ret.push_back(str.substr(start));
	return ret;
}

Mat PadSample(cv::Mat src) {
	//cvtColor(img, img, CV_RGB2GRAY);
	Mat img = src.clone();
	int h = img.rows;
	int w = img.cols;
	int target_h = 32; 
	int target_w = 128;
	float scale = (float)h / (float)target_h;
	int width = int((float)w / scale);
	Mat result;
	if (width > target_w)
	{
		resize(img, img, cv::Size(target_w, target_h));
		result = img.clone();
	}
	else
	{
		resize(img, img, cv::Size(width, target_h));
		Mat bg = Mat::zeros(target_h, target_w, CV_8U);
		cvtColor(bg, bg, CV_GRAY2BGR);
		int begin = (target_w - width) / 2;
		Mat roi = bg(cv::Rect(begin,0,width,target_h));
		img.copyTo(roi);
		result = bg.clone();
	}
	return result;
}

void main()
{
	cout << "loading..." << endl;
	string model_file = "julun/deploy.prototxt";
	string trained_file = "julun/julun.caffemodel";
	Classifier classifier(model_file, trained_file, BLANK_LABEL);
	cout << "loading finished" << endl;

	cout << "Please input the folder name : ";
	cin >> dir_path;
	Directory dir;
	fileNames = dir.GetListFiles(dir_path, "*.png", false);

	for (int i = 0; i < fileNames.size(); i++)
	{

		string fileName = fileNames[i];
		string fileFullName = dir_path + "/" + fileName;
		cout << "File name:" << fileName << endl;
		Mat src = imread(fileFullName);
		cv::Mat img = cv::imread(fileFullName, 1);
		img = PadSample(img);
		imshow("ori", img);
		CHECK(!img.empty()) << "Unable to decode image " << fileFullName;
		std::vector<int> predictions = classifier.Classify(img);

		std::cout << "Result:";
		string result = "";
		for (size_t i = 0; i < predictions.size(); ++i) 
		{
			int index = predictions[i];
			if (index != 45 && index != 46)
			{
				result += labelChar[predictions[i]];
			}
		}
		std::cout<< result << std::endl;

		waitKey(0);
	}
	system("pause");
}