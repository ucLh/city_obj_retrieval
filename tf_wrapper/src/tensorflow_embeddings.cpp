#include "tf_wrapper/tensorflow_embeddings.h"

#include <utility>

bool TensorFlowEmbeddings::set_input_output(
    std::vector<std::string> in_nodes, std::vector<std::string> out_nodes) {
  _input_node_names = std::move(in_nodes);
  _output_node_names = std::move(out_nodes);
  return true;
}

bool TensorFlowEmbeddings::normalize_image(cv::Mat &img) {
  double min, max;

  //    tf_aux::fast_resize_if_possible()
  //    cv::Scalar data = img.at<cv::Vec3b>(0,0);
  //    cv::minMaxLoc(img, &min, &max);
  //    img.convertTo(img, CV_32F, 1, 0); //TODO normalize it in a right way
  //    img = ((img - cv::Scalar(min, min, min)) / (max - min));
  //    img = (img * 2) - cv::Scalar(1);
  return true;
}

std::string TensorFlowEmbeddings::inference(const std::vector<cv::Mat> &imgs) {
  using namespace tensorflow;
  for (const cv::Mat &img : imgs) {
    if (!normalize_image(const_cast<cv::Mat &>(img))) {
      return "Fail to normalize images";
    }
  }

  if (!tf_aux::convert_mat_to_tensor_v2(imgs, _input_tensor)) {
    return "Fail to convert Mat to Tensor";
  }

#ifdef PHASEINPUT
  tensorflow::Tensor phase_tensor(tensorflow::DT_BOOL,
                                  tensorflow::TensorShape());
  phase_tensor.scalar<bool>()() = false;
  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {_input_node_name, input}, {"phase_train:0", phase_tensor}};
#else
  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {_input_node_names[0], _input_tensor}};
#endif

  _status = _session->Run(inputs, _output_node_names, {}, &_output_tensors);

  //    tf_aux::debug_output("NETWORK_STATUS", _status.ToString());
  return _status.ToString();
}

std::vector<std::vector<float>> TensorFlowEmbeddings::get_output_embeddings() {
  if (_output_tensors.empty() || !_is_loaded) {
    std::cerr << "Can't get output Embeddings" << std::endl;
    return {};
  }

  if (_out_embeddings.empty()) {
    const auto &output = _output_tensors[0];
    _out_embeddings = convert_tensor_to_vector(output);
  } else {
    _out_embeddings.clear();
    get_output_embeddings();
  }

  return _out_embeddings;
}

std::vector<std::vector<float>> TensorFlowEmbeddings::convert_tensor_to_vector(
    const tensorflow::Tensor &tensor) {
  const auto &temp_tensor = tensor.tensor<float, 2>();
  const auto &dims = tensor.shape();
  std::vector<float> temp_vec;

  //    TODO prealloc vector/array?
  std::vector<std::vector<float>> vec_embeddings;

  for (size_t batch_size = 0; batch_size < dims.dim_size(0); ++batch_size) {
    //        std::cout << batch_size << std::endl; //for debug
    for (size_t embedding_size = 0; embedding_size < dims.dim_size(1);
         ++embedding_size) {
      temp_vec.push_back(temp_tensor(batch_size, embedding_size));
    }

    vec_embeddings.push_back(temp_vec);
  }

  return vec_embeddings;
}

bool TensorFlowEmbeddings::set_gpu_number_preferred(int value) {
  TensorflowWrapperCore::set_gpu_number(value);
  const int gpu_num_value = TensorflowWrapperCore::get_gpu_number();
  if (gpu_num_value != value) {
    std::cerr << "GPU number was not set" << std::endl;
    return false;
  }

  return true;
}