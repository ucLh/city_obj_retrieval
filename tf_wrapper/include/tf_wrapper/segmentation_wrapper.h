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

  /// \brief
  /// \return
  bool process_images();

  bool process_images(const std::vector<std::string> &imgs_paths);

  bool process_images(const std::vector<cv::Mat> &images);

  bool prepare_for_inference(std::string config_path = "config.json");

  /// \brief
  /// \param resized
  /// \return
  std::vector<cv::Mat> get_indexed(); // =true

  /// \brief
  /// \param resized
  /// \return
  std::vector<cv::Mat> get_colored(); // =true

  /// \brief
  /// \param classes_to_mask
  /// \return
  std::vector<cv::Mat> get_masked(const std::set<int> &classes_to_mask);

  /// \brief Method for getting all visible devices that can handle computations
  /// \return
  std::string get_devices();

protected:
  bool _is_configured = false;
  cv::Size _img_des_size;
  std::vector<cv::Size> _img_orig_size;
  std::vector<cv::Mat> _imgs;
  std::unique_ptr<ISegmentationInterfaceHandler> inference_handler;
  std::unique_ptr<IDataBase> db_handler;
  std::vector<std::string> list_of_imgs;

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
