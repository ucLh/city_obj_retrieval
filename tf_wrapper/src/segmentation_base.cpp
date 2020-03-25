#include "tf_wrapper/segmentation_base.h"
#include "tf_wrapper/common/fs_handling.h"
#include "tf_wrapper/tensorflow_segmentator.h"
#include "tf_wrapper/wrapper_interfaces.h"

#include <utility>

SegmentationWrapperBase::SegmentationWrapperBase()
    : inference_handler(new SegmentationInferenceHandler),
      db_handler(new DataHandling) {}

bool SegmentationWrapperBase::set_images(
    const std::vector<std::string> &imgs_paths) {
  if (!_is_configured) {
    std::cerr << "You need to configure wrapper first!" << std::endl;
    return false;
  }
  fs_img::image_data_struct cur_img;
  for (const auto &img_path : imgs_paths) {
    cv::Mat img = fs_img::read_img(img_path);
    cur_img = fs_img::resize_img(img, _img_des_size);
    _imgs.emplace_back(std::move(cur_img.img_data));
    _img_orig_size.emplace_back(std::move(cur_img.orig_size));
  }
  return true;
}

bool SegmentationWrapperBase::process_images() {
  if (!_is_configured) {
    std::cerr << "You need to configure wrapper first!" << std::endl;
    return false;
  }
  inference_handler->clear_data(); /// Need to clear data that may be saved from
                                   /// previous launch
  for (unsigned long i = 0; i < _imgs.size(); ++i) {
    std::cout << "Wrapper Info:" << i + 1 << " of " << _imgs.size()
              << " was processed" << std::endl;
    inference_handler->inference({_imgs[i]});
  }
  return true;
}

std::vector<cv::Mat> SegmentationWrapperBase::get_indices(bool resized) {
  std::vector<cv::Mat> indices =
      inference_handler->get_output_segmentation_indices();
  if (resized) {
    for (auto i = 0; i != indices.size(); ++i) {
      cv::resize(indices[i], indices[i], _img_orig_size[i], 0, 0,
                 cv::INTER_LINEAR);
      cv::cvtColor(indices[i], indices[i], cv::COLOR_BGR2RGB);
    }
  }
  return indices;
}

std::vector<cv::Mat> SegmentationWrapperBase::get_colored(bool resized) {
  db_handler->load_colors();
  inference_handler->set_segmentation_colors(db_handler->get_colors());
  std::vector<cv::Mat> colored_indices =
      inference_handler->get_output_segmentation_colored();
  if (resized) {
    for (auto i = 0; i != colored_indices.size(); ++i) {
      cv::resize(colored_indices[i], colored_indices[i], _img_orig_size[i], 0,
                 0, cv::INTER_LINEAR);
      cv::cvtColor(colored_indices[i], colored_indices[i], cv::COLOR_BGR2RGB);
    }
  }
  return colored_indices;
}

/// For in-code parameter specification
bool SegmentationWrapperBase::configure_wrapper(
    const cv::Size &input_size, const std::string &colors_path,
    const std::string &pb_path, const std::string &input_node,
    const std::string &output_node) {
  _img_des_size = input_size;
  inference_handler->set_input_output({input_node}, {output_node});
  inference_handler->load(pb_path, input_node);
  db_handler->set_config_colors_path(colors_path);

  _is_configured = true;
  return true;
}

bool SegmentationWrapperBase::load_config(std::string config_path) {
  db_handler->set_config_path(std::move(config_path));
  if (!db_handler->load_config())
    return false;
  _img_des_size = db_handler->get_config_input_size();
  inference_handler->set_input_output({db_handler->get_config_input_node()},
                                      {db_handler->get_config_output_node()});
  inference_handler->load(db_handler->get_config_pb_path(),
                          db_handler->get_config_input_node());
  _is_configured = true;

  return true;
}

bool SegmentationWrapperBase::set_gpu(int value) {
  return inference_handler->set_gpu_number_preferred(value);
}
