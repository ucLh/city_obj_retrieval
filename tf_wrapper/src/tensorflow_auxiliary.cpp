#include "tf_wrapper/tensorflow_auxiliary.h"

std::vector<int> tf_aux::get_tensor_shape(tensorflow::Tensor &tensor) {
  std::vector<int> shape;
  int num_dimensions = tensor.shape().dims();
  for (int ii_dim = 0; ii_dim < num_dimensions; ii_dim++) {
    shape.push_back(tensor.shape().dim_size(ii_dim));
  }
  return shape;
}

bool tf_aux::convert_mat_to_tensor_v2(const std::vector<cv::Mat> &imgs,
                                      tensorflow::Tensor &tensor) {
  // We assume that images are already resized and normalized
  int height = imgs[0].size[0];
  int width = imgs[0].size[1];
  int batch_size = imgs.size();
  tensorflow::Tensor input_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({batch_size, height, width, 3}));

  auto input_tensor_mapped = input_tensor.tensor<float, 4>();

  for (size_t i = 0; i < batch_size; ++i) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        cv::Vec3b pixel = imgs[i].at<cv::Vec3b>(y, x);

        input_tensor_mapped(i, y, x, 0) = pixel.val[2]; // R
        input_tensor_mapped(i, y, x, 1) = pixel.val[1]; // G
        input_tensor_mapped(i, y, x, 2) = pixel.val[0]; // B
      }
    }
  }
  tensor = input_tensor;

  return true;
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
