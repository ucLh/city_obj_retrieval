#include "tf_wrapper/tensorflow_segmentator.h"

#include <utility>

std::vector<cv::Mat> TensorFlowSegmentator::get_output_segmentation_indices() {
  std::vector<cv::Mat> _out_data;
  if (!indices_.empty()) {
    indices_.clear();
  }
  for (const auto &output : out_tensors_vector_) {
    _out_data = std::move(TensorFlowSegmentator::convert_tensor_to_mat(output));
    indices_.insert(indices_.end(), _out_data.begin(), _out_data.end());
  }
  return indices_;
}

std::vector<cv::Mat> TensorFlowSegmentator::get_output_segmentation_colored() {
  if (!indices_.empty()) {
    indices_.clear();
  }
  std::vector<cv::Mat> _out_data;
  if (!colors_.empty()) {
    for (const auto &output : out_tensors_vector_) {
      _out_data =
          std::move(TensorFlowSegmentator::convert_tensor_to_mat(output));
      indices_.insert(indices_.end(), _out_data.begin(), _out_data.end());
    }
  } else {
    std::cerr << "Colors not set" << std::endl;
  }
  colors_.clear();
  return indices_;
}

std::vector<cv::Mat>
TensorFlowSegmentator::convert_tensor_to_mat(const tensorflow::Tensor &tensor) {

  if (is_loaded() && !output_tensors_.empty()) {
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
      if (colors_.empty()) {
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
              auto clrs(colors_[size_t(
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
  // PROFILE_BLOCK("inference time");

  if (!tf_aux::convert_mat_to_tensor_v2<tensorflow::DT_UINT8>(imgs,
                                                              input_tensor_)) {
    return "Fail to convert Mat to Tensor";
  }

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {input_node_names_[0], input_tensor_}};

  status_ = session_->Run(inputs, output_node_names_, {}, &output_tensors_);

  /// _output_tensors is a vector of tensors where each tensor represent every
  /// possible output from net taking 0'th out we are targeting tensor that
  /// contains output indices that we need

  out_tensors_vector_.emplace_back(std::move(output_tensors_[0]));
  TF_CHECK_OK(status_);
  //    tf_aux::DebugOutput("NETWORK_STATUS", status_.ToString());
  return status_.ToString();
}

bool TensorFlowSegmentator::set_segmentation_colors(
    std::vector<std::array<int, 3>> colors) {
  colors_ = std::move(colors);
  return true;
}

bool TensorFlowSegmentator::clear_data() {
  if (!out_tensors_vector_.empty())
    out_tensors_vector_.clear();
  if (!indices_.empty())
    indices_.clear();

  return true;
}
