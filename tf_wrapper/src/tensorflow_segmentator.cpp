#include "tf_wrapper/tensorflow_segmentator.h"

#include <utility>

std::vector<cv::Mat> TensorFlowSegmentator::get_output_segmentation_indices() {
  std::vector<cv::Mat> _out_data;
  if (_indices.empty()) {
    for (const auto &output : _out_tensors_vector) {
      _out_data =
          std::move(TensorFlowSegmentator::convert_tensor_to_mat(output));
      _indices.insert(_indices.begin(), _out_data.begin(), _out_data.end());
      //            this->_indices.emplace_back(TensorFlowSegmentator::convertTensorToMat(output));
    }
  } else
    _indices = {};
  return _indices;
}

std::vector<cv::Mat> TensorFlowSegmentator::get_output_segmentation_colored() {
  std::vector<cv::Mat> _out_data;
  if (_indices.empty() && !_colors.empty()) {
    for (const auto &output : _out_tensors_vector) {
      _out_data =
          std::move(TensorFlowSegmentator::convert_tensor_to_mat(output));
      _indices.insert(_indices.begin(), _out_data.begin(), _out_data.end());
    }
  } else {
    std::cerr << "Colors not set" << std::endl;
    _indices = {};
  }
  return _indices;
}

std::vector<cv::Mat>
TensorFlowSegmentator::convert_tensor_to_mat(const tensorflow::Tensor &tensor) {

  if (is_loaded() && !_output_tensors.empty()) {
    //        const auto &temp_tensor = tensor.tensor<tensorflow::int64, 4>();
    const auto &temp_tensor = tensor.tensor<tensorflow::int64, 3>();
    const auto &dims = tensor.shape();
    std::vector<cv::Mat> imgs(size_t(dims.dim_size(0)));

    for (size_t example = 0; example < imgs.size(); ++example) {
#if 0
      imgs[example] = cv::Mat(cv::Size_<int64>(dims.dim_size(1), dims.dim_size(2)), colors.size() ? CV_8UC3 : CV_8UC1);
#else
      imgs[example] = cv::Mat(
          cv::Size_<int64>(dims.dim_size(1), dims.dim_size(2)), CV_8UC3);
#endif
      if (_colors.empty()) {
#if 0
        imgs[example].forEach<uchar>([&](uchar& pixel, const int position[]) -> void {
                    pixel = uchar(temp_tensor(long(example), position[0], position[1], 0));
                });
#else
        imgs[example].forEach<cv::Vec3b>(
            [&](cv::Vec3b &pixel, const int position[]) -> void {
              //                    auto clrs = uchar(temp_tensor(long(example),
              //                    position[0], position[1], 0));
              auto clrs =
                  uchar(temp_tensor(long(example), position[0], position[1]));
              pixel = cv::Vec3b(cv::Vec3i{clrs, clrs, clrs});
            });
#endif
      } else
        imgs[example].forEach<cv::Vec3b>(
            [&](cv::Vec3b &pixel, const int position[]) -> void {
              //                    auto
              //                    clrs(this->_colors[size_t(temp_tensor(long(example),
              //                    position[0], position[1], 0))]);
              auto clrs(_colors[size_t(
                  temp_tensor(long(example), position[0], position[1]))]);
              pixel = cv::Vec3b(cv::Vec3i{clrs[0], clrs[1], clrs[2]});
            });
    }

    return imgs;
  } else
    return {};
}

std::string TensorFlowSegmentator::inference(const std::vector<cv::Mat> &imgs) {
  using namespace tensorflow;
  PROFILE_BLOCK("inference time");

  if (!tf_aux::convert_mat_to_tensor_v2<uint8_t>(imgs, _input_tensor,
                                                 tensorflow::DT_UINT8)) {
    return "Fail to convert Mat to Tensor";
  }

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {_input_node_names[0], _input_tensor}};

  _status = _session->Run(inputs, _output_node_names, {}, &_output_tensors);

  /// _output_tensors is a vector of tensors where each tensor represent every
  /// possible output from net taking 0'th out we targeting tensor that contains
  /// output indices that we need

  _out_tensors_vector.emplace_back(std::move(_output_tensors[0]));
  TF_CHECK_OK(_status);
  //    tf_aux::DebugOutput("NETWORK_STATUS", _status.ToString());
  return _status.ToString();
}

bool TensorFlowSegmentator::set_segmentation_colors(
    std::vector<std::array<int, 3>> colors) {
  _colors = std::move(colors);
  return true;
}

bool TensorFlowSegmentator::set_input_output(
    std::vector<std::string> in_nodes, std::vector<std::string> out_nodes) {
  _input_node_names = std::move(in_nodes);
  _output_node_names = std::move(out_nodes);
  return true;
}

bool TensorFlowSegmentator::clear_data() {
  if (!_out_tensors_vector.empty())
    _out_tensors_vector.clear();
  if (!_indices.empty())
    _indices.clear();

  return true;
}

bool TensorFlowSegmentator::set_gpu_number_preferred(int value) {
  TensorflowWrapperCore::set_gpu_number(value);
  const int gpu_num_value = TensorflowWrapperCore::get_gpu_number();
  if (gpu_num_value != value) {
    std::cerr << "GPU number was not set" << std::endl;
    return false;
  }

  return true;
}