#ifndef TF_WRAPPER_EMBEDDING_FS_HANDLING_H
#define TF_WRAPPER_EMBEDDING_FS_HANDLING_H

#include "fstream"
#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "string"
#include "vector"

#include "csv/csv.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "tf_wrapper/interfaces.h"

#define EXPERIMENTAL
#ifdef EXPERIMENTAL
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif
namespace fs_img {

struct image_data_struct {
  cv::Size orig_size;
  cv::Mat img_data;
};

cv::Mat read_img(const std::string &im_filename);

image_data_struct resize_img(const cv::Mat &orig_img, const cv::Size &size);

std::vector<std::string> list_imgs(const std::string &dir_path);
} // namespace fs_img

class DataHandling : public IDataBase {
public:
  DataHandling() = default;
  virtual ~DataHandling() = default;

  //    struct data_vec_entry
  //    {
  //        std::string filepath;
  //        std::vector<float> embedding;
  //    };
  struct config_data {
    cv::Size input_size;
    int top_n;
    std::string datafile_path;
    std::string imgs_path;
    std::string input_node;
    std::string pb_path;
    std::string output_node;
    std::string colors_path;
  };

  // struct with all config data
  config_data config;

  // important variable. It contains information of image paths and
  // corresponding embeddings.
  std::vector<data_vec_entry> data_vec_base;

  // vector with the colors for segmentation
  std::vector<std::array<int, 3>> colors;

  std::string config_path = "config.json";

  // TODO MOVE IT OUT
  bool load_database() override;
  bool load_config() override;
  bool add_json_entry(data_vec_entry new_data) override;

  bool add_error_entry(const std::string &act_class_in,
                       const std::string &act_path_in,
                       const std::string &expected_class_in) override;

  /// In case you want specific config to be used
  /// \param path to config
  /// \return if custom config is used
  bool set_config_path(std::string path) override;

  bool load_colors() override;

  cv::Size get_config_input_size() override;

  std::string get_config_input_node() override;

  std::string get_config_output_node() override;

  std::string get_config_pb_path() override;

  std::string get_config_imgs_path() override;

  int get_config_top_n() override;

  std::vector<std::array<int, 3>> get_colors() override;

  bool set_data_vec_base(const std::vector<data_vec_entry> &base) override;

  bool set_config_input_size(const cv::Size &size) override;

  bool set_config_input_node(const std::string &input_node) override;

  bool set_config_output_node(const std::string &output_node) override;

  bool set_config_pb_path(const std::string &embed_pb_path) override;

  bool set_config_colors_path(const std::string &colors_path) override;

  bool set_config_imgs_path(const std::string &imgs_path) override;

  std::vector<DataHandling::data_vec_entry> get_data_vec_base() override;

  void
  add_element_to_data_vec_base(DataHandling::data_vec_entry &entry) override;

protected:
  std::fstream imgs_datafile_;
  std::fstream config_datafile_;
  std::fstream errors_datafile_;

  bool open_datafile();
  bool open_config();
  bool open_error_datafile();
  static std::string try_parse_json_member(rapidjson::Document &doc,
                                           const std::string &name,
                                           const std::string &default_val = "");
};

#endif // TF_WRAPPER_EMBEDDING_FS_HANDLING_H
