#include "tf_wrapper/segmentation_wrapper.h"
#include "tf_wrapper/common/fs_handling.h"
#include "tf_wrapper/inference_handlers.h"
#include "tf_wrapper/tensorflow_segmentator.h"

#include <utility>

SegmentationWrapper::SegmentationWrapper() {
  inference_handler_ = std::make_unique<SegmentationInferenceHandler>();
  db_handler_ = std::make_unique<DataHandling>();
}

bool SegmentationWrapper::load_config(std::string config_path) {
  db_handler_->set_config_path(std::move(config_path));
  if (!db_handler_->load_config()) {
    std::cerr << "Can't load config!" << std::endl;
    return false;
  }
  img_des_size_ = db_handler_->get_config_input_size();
  inference_handler_->set_input_output({db_handler_->get_config_input_node()},
                                      {db_handler_->get_config_output_node()});
  inference_handler_->load(db_handler_->get_config_pb_path(),
                           db_handler_->get_config_input_node());
  is_configured_ = true;

  return true;
}

bool SegmentationWrapper::prepare_for_inference(std::string config_path) {
  load_config(std::move(config_path));
  list_of_imgs_ = fs_img::list_imgs(db_handler_->get_config_imgs_path());
  set_images(list_of_imgs_);
}

bool SegmentationWrapper::set_images(
    const std::vector<std::string> &imgs_paths) {
  if (!is_configured_) {
    std::cerr << "You need to configure wrapper first!" << std::endl;
    return false;
  }
  if (!imgs_.empty()) { /// Delete previously processed images
    imgs_.clear();
    img_orig_size_.clear();
  }
  cv::Mat img;
  for (const auto &img_path : imgs_paths) {
    img = fs_img::read_img(img_path);
    img_orig_size_.emplace_back(img.size());
    imgs_.emplace_back(std::move(img));
  }
  return true;
}

bool SegmentationWrapper::process_images() { return process_images(imgs_); }

bool SegmentationWrapper::process_images(
    const std::vector<std::string> &imgs_paths) {
  set_images(imgs_paths);
  process_images();
}

bool SegmentationWrapper::process_images(const std::vector<cv::Mat> &images) {
  if (!is_configured_) {
    std::cerr << "You need to configure wrapper first!" << std::endl;
    return false;
  }
  fs_img::image_data_struct resized_img;
  inference_handler_->clear_data(); /// Need to clear data that may be saved from
                                   /// previous launch
  for (unsigned long i = 0; i < images.size(); ++i) {
    resized_img = fs_img::resize_img(images[i], img_des_size_);
    inference_handler_->inference({resized_img.img_data});
    std::cout << "Wrapper Info: " << i + 1 << " of " << images.size()
              << " was processed" << "\r" << std::flush;
  }
  std::cout << std::endl;
  return true;
}

std::vector<cv::Mat> SegmentationWrapper::get_indexed() {
  std::vector<cv::Mat> indices =
      inference_handler_->get_output_segmentation_indices();
  for (auto i = 0; i != indices.size(); ++i) {
    cv::resize(indices[i], indices[i], img_orig_size_[i], 0, 0,
               cv::INTER_LINEAR);
    cv::cvtColor(indices[i], indices[i], cv::COLOR_BGR2RGB);
  }
  return indices;
}

std::vector<cv::Mat> SegmentationWrapper::get_colored() {
  db_handler_->load_colors();
  inference_handler_->set_segmentation_colors(db_handler_->get_colors());
  std::vector<cv::Mat> colored_indices =
      inference_handler_->get_output_segmentation_colored();
  for (auto i = 0; i != colored_indices.size(); ++i) {
    cv::resize(colored_indices[i], colored_indices[i], img_orig_size_[i], 0, 0,
               cv::INTER_LINEAR);
    cv::cvtColor(colored_indices[i], colored_indices[i], cv::COLOR_BGR2RGB);
  }
  return colored_indices;
}

std::vector<cv::Mat>
SegmentationWrapper::get_masked(const std::set<int> &classes_to_mask) {
  std::vector<cv::Mat> indices = SegmentationWrapper::get_indexed();
  std::vector<cv::Mat> result_imgs;
  for (unsigned long i = 0; i < indices.size(); ++i) {
    auto cur_img = imgs_[i];
    for (int x = 0; x < indices[i].rows; ++x) {
      for (int y = 0; y < indices[i].cols; ++y) {
        auto pixel = indices[i].at<cv::Vec3b>(x, y)[0];
        if (classes_to_mask.count(pixel) != 0) {
          auto &color = cur_img.at<cv::Vec3b>(x, y);
          color[0] = 0;
          color[1] = 0;
          color[2] = 0;
        }
      }
    }
    result_imgs.emplace_back(cur_img);
  }
  return result_imgs;
}

bool SegmentationWrapper::set_gpu(int value) {
  return inference_handler_->set_gpu_number_preferred(value);
}
