#ifndef TF_WRAPPER_METRICS_BASE_H
#define TF_WRAPPER_METRICS_BASE_H

#include "tf_wrapper/embeddings_wrapper.h"
#include <algorithm>

class MetricsBase : public EmbeddingsWrapper {
public:
  MetricsBase() {
    //        this->topN = 5;
    db_handler_->set_config_path("embed_config.json");
  };

  ~MetricsBase() = default;

  /// using accuracy as metrics
  /// \param queries_path path to test image
  /// \param top_N_classes number of plausible classes
  /// \return value of accuracy
  float get_metrics(std::string &queries_path, int top_N_classes = 4,
                    bool use_segmentation = false);

  //  std::vector<EmbeddingsWrapper::distance>
  //  inference_and_matching(std::string img_path) override;

  struct testimg_entry {
    std::string img_path;
    std::string img_class;
    std::vector<std::string> img_classes_proposed;
    cv::Mat img;
    float distance;
    bool is_correct;
    // TODO add correction distance
  };

protected:
  std::string queries_path_;
  std::vector<testimg_entry> testimg_vector;

  bool prepare_for_inference(std::string config_path) override;

  std::vector<std::string> choose_classes(
      const std::vector<EmbeddingsWrapper::distance> &matched_images_list,
      testimg_entry &test_img, unsigned int top_N_classes,
      std::string &queries_path, std::string &series_path);

  double calculate_average_precision(
      const std::vector<EmbeddingsWrapper::distance> &matched_images_list,
      const std::string &target_class_name);
};

#endif // TF_WRAPPER_METRICS_BASE_H
