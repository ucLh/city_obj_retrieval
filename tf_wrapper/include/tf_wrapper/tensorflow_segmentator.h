#ifndef TF_WRAPPER_SEGMENTATION_TENSORFLOW_SEGMENTATOR_H
#define TF_WRAPPER_SEGMENTATION_TENSORFLOW_SEGMENTATOR_H

#include "opencv2/imgproc/imgproc.hpp"
#include "tensorflow_base.h"
#include "tensorflow_wrapper_core.h"

#include "tensorflow_base.h"
#include "tensorflow_wrapper_core.h"
#include <cmath>

class TensorFlowSegmentator : public TensorFlowWrapperCore {
public:
  TensorFlowSegmentator() { colors_ = {}; };
  ~TensorFlowSegmentator() override = default;

  std::string inference(const std::vector<cv::Mat> &imgs) override;

  virtual std::vector<cv::Mat> get_output_segmentation_indices();

  virtual std::vector<cv::Mat> get_output_segmentation_colored();

  virtual bool set_segmentation_colors(std::vector<std::array<int, 3>> colors);

  virtual bool clear_data();

protected:
  std::vector<cv::Mat> convert_tensor_to_mat(const tensorflow::Tensor &tensor);

  std::vector<std::array<int, 3>> colors_;
  tensorflow::Status status_;
  tensorflow::Tensor input_tensor_;

  std::vector<tensorflow::Tensor> out_tensors_vector_;
  std::vector<cv::Mat> indices_;
};

#endif // TF_WRAPPER_SEGMENTATION_TENSORFLOW_SEGMENTATOR_H
