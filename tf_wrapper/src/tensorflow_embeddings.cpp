#include "tf_wrapper/tensorflow_embeddings.h"

#include <utility>

std::string TensorFlowEmbeddings::inference(const std::vector<cv::Mat> &imgs) {
  using namespace tensorflow;
  // PROFILE_BLOCK("inference time");

  if (!tf_aux::convert_mat_to_tensor_v2<tensorflow::DT_FLOAT>(imgs,
                                                              _input_tensor)) {
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
