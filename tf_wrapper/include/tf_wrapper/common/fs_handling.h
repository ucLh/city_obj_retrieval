#ifndef TF_WRAPPER_EMBEDDING_FS_HANDLING_H
#define TF_WRAPPER_EMBEDDING_FS_HANDLING_H

#include "fstream"
#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "string"
#include "vector"

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

cv::Mat read_img(const std::string &im_filename, const cv::Size &size);

std::vector<std::string> list_imgs(const std::string &dir_path);
} // namespace fs_img

class DataHandling : public DBInterface {
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
  };

  // struct with all config data
  config_data config;

  // important variable. It contains information of image paths and
  // corresponding embeddings.
  std::vector<data_vec_entry> data_vec_base;

  std::string config_path = "config.json";

  // TODO MOVE IT OUT
  bool load_database();
  bool load_config();
  bool add_json_entry(data_vec_entry new_data);

  bool add_error_entry(const std::string &act_class_in,
                       const std::string &act_path_in,
                       const std::string &expected_class_in);

  /// In case you want specific config to be used
  /// \param path to config
  /// \return if custom config is used
  bool set_config_path(std::string path) override;

  cv::Size get_config_input_size();

  std::string get_config_input_node();

  std::string get_config_output_node();

  std::string get_config_pb_path();

  std::string get_config_imgs_path();

  int get_config_top_n();

  bool set_data_vec_base(const std::vector<data_vec_entry> &base);

  bool set_config_input_size(const cv::Size &size);

  bool set_config_input_node(const std::string &input_node);

  bool set_config_output_node(const std::string &output_node);

  bool set_config_pb_path(const std::string &pb_path);

  std::vector<DataHandling::data_vec_entry> get_data_vec_base();

  void add_element_to_data_vec_base(DataHandling::data_vec_entry &entry);

protected:
  std::fstream imgs_datafile;
  std::fstream config_datafile;
  std::fstream errors_datafile;

  bool open_datafile();
  bool open_config();
  bool open_error_datafile();
};

#endif // TF_WRAPPER_EMBEDDING_FS_HANDLING_H
