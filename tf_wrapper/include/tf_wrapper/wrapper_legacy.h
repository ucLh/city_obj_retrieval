#ifndef TF_WRAPPER_EMBEDDING_WRAPPER_LEGACY_H
#define TF_WRAPPER_EMBEDDING_WRAPPER_LEGACY_H

#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "tensorflow_auxiliary.h"

#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/public/session.h"

namespace wrapper_legacy {

/// \brief convertTensorToMat Converts batch of Mats to Tensor format
/// \param imgs Vector of images (batch) of cv::Mat format, they must be uchar
/// type \param height Output Tensor height \param width Output Tensor width
/// \param depth Output Tensor depth
/// \return Ready to inference Tensor with batch of images
template <tensorflow::DataType T>
tensorflow::Tensor convert_mat_to_tensor(const std::vector<cv::Mat> &imgs,
                                         int height, int width, int depth,
                                         bool normalize,
                                         const std::vector<float> &mean) {

  using namespace tensorflow;

  if (mean.size() != depth) {
    std::cerr << "convertMatToTensor: wrong mean or channel size!" << std::endl;
    return Tensor();
  }

  Tensor input(T, TensorShape({imgs.size(), height, width, depth}));

  using POD_type = typename tensorflow::EnumToDataType<T>::Type;
  auto input_tensor_mapped = input.tensor<POD_type, 4>();
  //        auto input_tensor_mapped = input.tensor<float, 4>();

  const auto batch_size = imgs.size();
  for (size_t i = 0; i < batch_size; ++i) {
    cv::Mat img;
#ifdef TFDEBUG
    {
      PROFILE_BLOCK("resize img");
#endif
      tf_aux::fast_resize_if_possible(imgs[i], &img, cv::Size(width, height));
#ifdef TFDEBUG
    }
#endif

#ifdef TFDEBUG
    {
      PROFILE_BLOCK("convert img");
#endif

      img.forEach<cv::Vec3b>(
          [&](cv::Vec3b &pixel, const int position[]) -> void {
            for (short c = 0; c < 3; ++c) {
              POD_type val(pixel[c]);
              val -= mean[c];
              if (normalize) {
                val /= 255.0;
                val -= 0.5;
                val *= 2.0;
              }
              // "2 - c" performs cv::COLOR_BGR2RGB
              input_tensor_mapped(i, position[0], position[1], 2 - c) = val;
            }
          });
#ifdef TFDEBUG
    }
#endif
  }
  return input;
}

} // namespace wrapper_legacy

#endif // TF_WRAPPER_EMBEDDING_WRAPPER_LEGACY_H
