#ifndef TF_WRAPPER_INTERFACES_H
#define TF_WRAPPER_INTERFACES_H

#include <memory>
#include <opencv2/core/mat.hpp>
#include <vector>

class IInferenceHandler {
public:
  virtual bool set_gpu_number_preferred(int value) = 0;

  virtual bool set_input_output(std::vector<std::string> in_nodes,
                                std::vector<std::string> out_nodes) = 0;

  virtual bool load(const std::string &filename,
                    const std::string &inputNodeName) = 0;

  virtual std::string inference(const std::vector<cv::Mat> &imgs) = 0;

  virtual std::string get_visible_devices() = 0;

  virtual void clear_data() = 0;
};

class IEmbeddingsInferenceHandler : public IInferenceHandler {
public:
  virtual bool is_loaded() = 0;

  virtual std::vector<std::vector<float>> get_output_embeddings() = 0;
};

class ISegmentationInterfaceHandler : public IInferenceHandler {
public:
  virtual bool
  set_segmentation_colors(std::vector<std::array<int, 3>> colors) = 0;

  virtual std::vector<cv::Mat> get_output_segmentation_indices() = 0;

  virtual std::vector<cv::Mat> get_output_segmentation_colored() = 0;
};

class DBInterface {
public:
  struct data_vec_entry {
    std::string filepath;
    std::vector<float> embedding;
  };

  virtual bool set_config_path(std::string path) = 0;

  virtual bool load_config() = 0;

  virtual bool load_database() = 0;

  virtual bool load_colors() = 0;

  virtual bool add_json_entry(data_vec_entry new_data) = 0;

  virtual void add_element_to_data_vec_base(data_vec_entry &entry) = 0;

  virtual bool add_error_entry(const std::string &act_class_in,
                               const std::string &act_path_in,
                               const std::string &expected_class_in) = 0;

  virtual std::vector<data_vec_entry> get_data_vec_base() = 0;

  virtual cv::Size get_config_input_size() = 0;

  virtual std::string get_config_input_node() = 0;

  virtual std::string get_config_output_node() = 0;

  virtual std::string get_config_pb_path() = 0;

  virtual std::string get_config_imgs_path() = 0;

  virtual int get_config_top_n() = 0;

  virtual std::vector<std::array<int, 3>> get_colors() = 0;

  virtual bool set_data_vec_base(const std::vector<data_vec_entry> &base) = 0;

  virtual bool set_config_input_size(const cv::Size &size) = 0;

  virtual bool set_config_input_node(const std::string &input_node) = 0;

  virtual bool set_config_output_node(const std::string &output_node) = 0;

  virtual bool set_config_pb_path(const std::string &embed_pb_path) = 0;

  virtual bool set_config_colors_path(const std::string &colors_path) = 0;
};
#endif // TF_WRAPPER_INTERFACES_H
