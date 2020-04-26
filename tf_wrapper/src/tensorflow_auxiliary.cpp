#include "tf_wrapper/tensorflow_auxiliary.h"

std::vector<int> tf_aux::get_tensor_shape(tensorflow::Tensor &tensor) {
  std::vector<int> shape;
  int num_dimensions = tensor.shape().dims();
  for (int ii_dim = 0; ii_dim < num_dimensions; ii_dim++) {
    shape.push_back(tensor.shape().dim_size(ii_dim));
  }
  return shape;
}

bool tf_aux::fast_resize_if_possible(const cv::Mat &in, cv::Mat *dist,
                                     const cv::Size &size) {
  if (in.size() == size) {
    debug_output("sizes matches", std::to_string(in.cols) + "x" +
                                      std::to_string(in.rows) + "; " +
                                      std::to_string(size.width) + "x" +
                                      std::to_string(size.height));
    in.copyTo(*dist);
    return true;
  }

  cv::resize(in, *dist, size, 0, 0, cv::INTER_LINEAR);
  return true;
}
