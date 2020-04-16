#ifndef TF_WRAPPER_INFERENCE_HANDLERS_H
#define TF_WRAPPER_INFERENCE_HANDLERS_H

#include "interfaces.h"
#include "tensorflow_embeddings.h"
#include "tensorflow_segmentator.h"

class EmbeddingsInferenceHandler : public IEmbeddingsInferenceHandler {
public:
  //*
  bool set_gpu_number_preferred(int value) override {
    return embed.set_gpu_number_preferred(value);
  }
  //
  bool load(const std::string &filename,
            const std::string &inputNodeName) override {
    return embed.load(filename, inputNodeName);
  }
  //*
  bool set_input_output(std::vector<std::string> in_nodes,
                        std::vector<std::string> out_nodes) override {
    return embed.set_input_output(in_nodes, out_nodes);
  }

  std::string inference(const std::vector<cv::Mat> &imgs) override {
    return embed.inference(imgs);
  }

  //
  std::string get_visible_devices() override {
    return embed.get_visible_devices();
  } // Does not really work yet

  bool is_loaded() override { return embed.is_loaded(); }

  std::vector<std::vector<float>> get_output_embeddings() override {
    return embed.get_output_embeddings();
  }

  void clear_data() override { embed.clear_session(); }

private:
  TensorFlowEmbeddings embed;
};

class SegmentationInferenceHandler : public ISegmentationInterfaceHandler {
public:
  bool set_gpu_number_preferred(int value) override {
    return segm.set_gpu_number_preferred(value);
  }

  bool load(const std::string &filename,
            const std::string &inputNodeName) override {
    return segm.load(filename, inputNodeName);
  }

  bool set_input_output(std::vector<std::string> in_nodes,
                        std::vector<std::string> out_nodes) override {
    return segm.set_input_output(in_nodes, out_nodes);
  }

  std::string inference(const std::vector<cv::Mat> &imgs) override {
    return segm.inference(imgs);
  }

  std::string get_visible_devices() override {
    return segm.get_visible_devices();
  } // Does not really work yet

  void clear_data() override { segm.clear_data(); }

  bool
  set_segmentation_colors(std::vector<std::array<int, 3>> colors) override {
    return segm.set_segmentation_colors(colors);
  }

  std::vector<cv::Mat> get_output_segmentation_indices() override {
    return segm.get_output_segmentation_indices();
  }

  std::vector<cv::Mat> get_output_segmentation_colored() override {
    return segm.get_output_segmentation_colored();
  }

private:
  TensorFlowSegmentator segm;
};

#endif // TF_WRAPPER_INFERENCE_HANDLERS_H
