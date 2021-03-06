#ifndef TF_WRAPPER_EMBEDDING_TENSORFLOW_EMBEDDINGS_H
#define TF_WRAPPER_EMBEDDING_TENSORFLOW_EMBEDDINGS_H

#include "opencv2/imgproc/imgproc.hpp"
#include "tensorflow_base.h"
#include "tensorflow_wrapper_core.h"

#include <cmath>

class TensorFlowEmbeddings : public TensorFlowWrapperCore {
public:
  TensorFlowEmbeddings() = default;
  ~TensorFlowEmbeddings() override = default;

  // int batch_size;

  /// \brief function for inferencing vector of input images
  /// \param imgs is vector of images
  /// \return status message
  std::string inference(const std::vector<cv::Mat> &imgs) override;

  /// \brief provides vector of output embeddings after inference
  /// \return vector of output embeddings after inference
  std::vector<std::vector<float>> get_output_embeddings();

protected:
  /// \brief function to convert output tensor of embeddings to vector of
  /// embeddings
  /// \param tensor
  /// \return vector of embeddings
  static std::vector<std::vector<float>>
  convert_tensor_to_vector(const tensorflow::Tensor &tensor);

  tensorflow::Status status_;
  tensorflow::Tensor input_tensor_;
  std::vector<std::vector<float>> out_embeddings_;
};

#endif // TF_WRAPPER_EMBEDDING_TENSORFLOW_EMBEDDINGS_H
