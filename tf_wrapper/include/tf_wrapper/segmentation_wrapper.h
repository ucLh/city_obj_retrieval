#ifndef TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H
#define TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H

#include "interfaces.h"
#include <set>

class SegmentationWrapper {
public:
  SegmentationWrapper();

  ~SegmentationWrapper() = default;

  /// \brief Method for setting number of GPU to make calculations on.
  /// \param value
  bool set_gpu(int value);

  /// \brief Main method for processing images with a segmentation neural
  /// network
  bool process_images();

  bool process_images(const std::vector<std::string> &imgs_paths);

  bool process_images(const std::vector<cv::Mat> &images);

  /// \brief Main method used for reading images in directory and adding them to
  /// the database
  bool prepare_for_inference(std::string config_path = "config.json");

  /// \brief Provides indexes of an image after it has been processed
  std::vector<cv::Mat> get_indexed(); // =true

  /// \brief Provides color mask of an image after it has been processed
  std::vector<cv::Mat> get_colored(); // =true

  /// \brief Provides masked image after it has been processed
  /// \param classes_to_mask a set of integers corresponding to classes in
  /// classes.csv (the classes file that is specified in the config)
  std::vector<cv::Mat> get_masked(const std::set<int> &classes_to_mask);

  /// \brief Method for getting all visible devices that can handle computations
  /// \return
  std::string get_devices();

protected:
  bool is_configured_ = false;
  cv::Size img_des_size_;
  std::vector<cv::Size> img_orig_size_;
  std::vector<cv::Mat> imgs_;
  std::unique_ptr<ISegmentationInterfaceHandler> inference_handler_;
  std::unique_ptr<IDataBase> db_handler_;
  std::vector<std::string> list_of_imgs_;

  /// \brief Method for configuring wrapper if config need to be loaded from
  /// file
  /// \param config_path is a path to .json file with config to wrapper
  /// \return
  bool load_config(std::string config_path); // ="config.json"

  /// \brief Method for setting image paths to process
  /// \param imgs_paths
  /// \return
  bool set_images(
      const std::vector<std::string> &imgs_paths); // opt for future_batch
};

#endif // TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H
