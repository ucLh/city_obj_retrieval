#ifndef TF_WRAPPER_EMBEDDING_WRAPPER_BASE_H
#define TF_WRAPPER_EMBEDDING_WRAPPER_BASE_H

#include "interfaces.h"

namespace EmbeddingMatching {
static float calc_distance_euclid(std::vector<float> base,
                                  std::vector<float> target);

float calc_distance_cosine(std::vector<float> base, std::vector<float> target);
} // namespace EmbeddingMatching

class EmbeddingsWrapper {
public:
  EmbeddingsWrapper();

  ~EmbeddingsWrapper() = default;

  struct distance {
    float dist;
    std::string path;
  };

  unsigned int topN;

  /// \brief Main method used for reading images in directory and adding the to
  /// the database
  bool prepare_for_inference(std::string config_path);

  /// \brief Main method used for matching passed image with images that already
  /// in database
  /// \param img_path passed image
  /// \return vector of distance between passed image and db images
  std::vector<EmbeddingsWrapper::distance>
  inference_and_matching(const std::string &img_path);

  /// \brief Same function as the above one. Needs a cv::Mat img instead of
  /// a path to file
  /// \param cv::Mat img
  std::vector<EmbeddingsWrapper::distance> inference_and_matching(cv::Mat img);

protected:
  bool is_configured_ = false;
  std::unique_ptr<IDataBase> db_handler_;
  std::unique_ptr<IEmbeddingsInferenceHandler> inference_handler_;
  std::vector<std::string> list_of_imgs_;
  std::vector<EmbeddingsWrapper::distance> distances_;

  /// \brief Method for loading config
  /// \param config_path path to config file
  bool load_config(std::string config_path);

  /// \brief Method for embeddings matching that is called from
  /// inference_and_matching method
  bool matching(const std::vector<IDataBase::data_vec_entry> &base,
                std::vector<float> &target);

  bool add_updates();
  bool check_for_updates();
};

#endif // TF_WRAPPER_EMBEDDING_WRAPPER_BASE_H
