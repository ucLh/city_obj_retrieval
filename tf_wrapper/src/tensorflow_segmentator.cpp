#include "tf_wrapper/tensorflow_segmentator.h"

#include <utility>

// TODO Main objective is ti remember about Batch

std::vector<cv::Mat> TensorFlowSegmentator::get_output_segmentation_indices() {
  if (_indices.empty()) {
    const auto output = _output_tensors[0];
    this->_indices = TensorFlowSegmentator::convert_tensor_to_mat(output);
  } else
    this->_indices = {};
  return this->_indices;
}

std::vector<cv::Mat> TensorFlowSegmentator::get_output_segmentation_colored() {
  if (this->_indices.empty() && !this->_colors.empty()) {
    const auto output = _output_tensors[0];
    this->_indices = TensorFlowSegmentator::convert_tensor_to_mat(output);
  } else {
    std::cerr << "Colors not set" << std::endl;
    this->_indices = {};
  }
  return this->_indices;
}

// bool TensorFlowSegmentator::normalize_image(cv::Mat &img) {
//  double min, max;
//  cv::Scalar data = img.at<cv::Vec3b>(0, 0);
//  cv::minMaxLoc(img, &min, &max);
//  img.convertTo(img, CV_32F, 1, 0); // TODO normalize it in a right way
//  img = ((img - cv::Scalar(min, min, min)) / (max - min));
//  img = (img * 2) - cv::Scalar(1);
//
//  return true;
//}

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
      if (this->_colors.empty()) {
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
              auto clrs(this->_colors[size_t(
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

  //    for (const cv::Mat &img : imgs) {
  //        if(!normalize_image(const_cast<cv::Mat &>(img))){
  //            return "Fail to normalize images";
  //        }
  //    }
  //    this->_input_tensor =
  //    wrapper_legacy::convertMatToTensor<tensorflow::DT_UINT8>(imgs, 256, 256,
  //    3, false, {0, 0, 0});
  if (!tf_aux::convert_mat_to_tensor_v2<uint8_t>(imgs, _input_tensor,
                                                 tensorflow::DT_UINT8)) {
    return "Fail to convert Mat to Tensor";
  }

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {_input_node_names[0], this->_input_tensor}};

  _status = _session->Run(inputs, _output_node_names, {}, &_output_tensors);

  TF_CHECK_OK(_status);
  //    tf_aux::DebugOutput("NETWORK_STATUS", _status.ToString());
  return _status.ToString();
}

bool TensorFlowSegmentator::set_segmentation_colors(
    std::vector<std::array<int, 3>> colors) {
  this->_colors = std::move(colors);
  return true;
}

bool TensorFlowSegmentator::set_input_output(
    std::vector<std::string> in_nodes, std::vector<std::string> out_nodes) {
  this->_input_node_names = std::move(in_nodes);
  this->_output_node_names = std::move(out_nodes);
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