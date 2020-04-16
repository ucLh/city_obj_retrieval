#ifndef TF_WRAPPER_EMBEDDING_TENSORFLOW_EMBEDDINGS_H
#define TF_WRAPPER_EMBEDDING_TENSORFLOW_EMBEDDINGS_H

#include "opencv2/imgproc/imgproc.hpp"
#include "tensorflow_base.h"
#include "tensorflow_wrapper_core.h"

#include <cmath>

class TensorFlowEmbeddings : public TensorflowWrapperCore {
public:
  TensorFlowEmbeddings() = default;
  ~TensorFlowEmbeddings() override = default;

  int batch_size;

  bool set_input_output(std::vector<std::string> in_nodes,
                        std::vector<std::string> out_nodes);

  /// \brief function for inferencing vector of input images
  /// \param imgs is vector of images
  /// \return status message
  std::string inference(const std::vector<cv::Mat> &imgs) override;

  /// \brief
  /// \return vector of output embeddings after inference
  std::vector<std::vector<float>> get_output_embeddings();

  /// \brief function to convert output tensor of embeddings to vector of
  /// embeddings \param tensor \return Vector of embeddings
  static std::vector<std::vector<float>>
  convert_tensor_to_vector(const tensorflow::Tensor &tensor);

  bool set_gpu_number_preferred(int value);

protected:
  tensorflow::Status _status;
  tensorflow::Tensor _input_tensor;
  std::vector<std::vector<float>> _out_embeddings;
};

#endif // TF_WRAPPER_EMBEDDING_TENSORFLOW_EMBEDDINGS_H
