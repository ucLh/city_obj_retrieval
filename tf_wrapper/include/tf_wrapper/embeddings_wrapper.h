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

  /// \brief main method used for reading images in directory and adding the to
  /// th database
  /// \return
  virtual bool prepare_for_inference(std::string config_path);

  /// \brief main method used for matching passed image with images that already
  /// in database
  /// \param img_path passed image
  /// \return vector of distance between passed image and db images
  virtual std::vector<EmbeddingsWrapper::distance>
  inference_and_matching(std::string img_path);

protected:
  bool is_configured_ = false;
  std::unique_ptr<IDataBase> db_handler_;
  std::unique_ptr<IEmbeddingsInferenceHandler> inference_handler_;
  std::vector<std::string> list_of_imgs_;
  std::vector<EmbeddingsWrapper::distance> distances_;

  /// \brief method for loading config
  /// \param config_path path to config file
  /// \return
  bool load_config(std::string config_path);

  bool matching(const std::vector<IDataBase::data_vec_entry> &base,
                 std::vector<float> &target);

  bool add_updates();
  bool check_for_updates();
};

#endif // TF_WRAPPER_EMBEDDING_WRAPPER_BASE_H
