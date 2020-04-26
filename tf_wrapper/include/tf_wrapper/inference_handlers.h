#ifndef TF_WRAPPER_INFERENCE_HANDLERS_H
#define TF_WRAPPER_INFERENCE_HANDLERS_H

#include "interfaces.h"
#include "tensorflow_embeddings.h"
#include "tensorflow_segmentator.h"

class EmbeddingsInferenceHandler : public IEmbeddingsInferenceHandler {
public:
  bool set_gpu_number_preferred(int value) override {
    return embed_.set_gpu_number_preferred(value);
  }

  bool load(const std::string &filename,
            const std::string &inputNodeName) override {
    return embed_.load(filename, inputNodeName);
  }

  bool set_input_output(std::vector<std::string> in_nodes,
                        std::vector<std::string> out_nodes) override {
    return embed_.set_input_output(in_nodes, out_nodes);
  }

  std::string inference(const std::vector<cv::Mat> &imgs) override {
    return embed_.inference(imgs);
  }

  std::string get_visible_devices() override {
    return embed_.get_visible_devices();
  } // Does not really work yet

  bool is_loaded() override { return embed_.is_loaded(); }

  std::vector<std::vector<float>> get_output_embeddings() override {
    return embed_.get_output_embeddings();
  }

  void clear_data() override { embed_.clear_session(); }

private:
  TensorFlowEmbeddings embed_;
};

class SegmentationInferenceHandler : public ISegmentationInterfaceHandler {
public:
  bool set_gpu_number_preferred(int value) override {
    return segm_.set_gpu_number_preferred(value);
  }

  bool load(const std::string &filename,
            const std::string &inputNodeName) override {
    return segm_.load(filename, inputNodeName);
  }

  bool set_input_output(std::vector<std::string> in_nodes,
                        std::vector<std::string> out_nodes) override {
    return segm_.set_input_output(in_nodes, out_nodes);
  }

  std::string inference(const std::vector<cv::Mat> &imgs) override {
    return segm_.inference(imgs);
  }

  std::string get_visible_devices() override {
    return segm_.get_visible_devices();
  } // Does not really work yet

  void clear_data() override { segm_.clear_data(); }

  bool
  set_segmentation_colors(std::vector<std::array<int, 3>> colors) override {
    return segm_.set_segmentation_colors(colors);
  }

  std::vector<cv::Mat> get_output_segmentation_indices() override {
    return segm_.get_output_segmentation_indices();
  }

  std::vector<cv::Mat> get_output_segmentation_colored() override {
    return segm_.get_output_segmentation_colored();
  }

private:
  TensorFlowSegmentator segm_;
};

#endif // TF_WRAPPER_INFERENCE_HANDLERS_H
