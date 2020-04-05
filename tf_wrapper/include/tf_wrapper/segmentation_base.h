#ifndef TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H
#define TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H

#include "interfaces.h"

class SegmentationWrapperBase {
public:
  SegmentationWrapperBase();

  ~SegmentationWrapperBase() = default;

  /// \brief Method for setting number of GPU to make calculations on.
  /// \param value
  bool set_gpu(int value);

  /// \brief
  /// \param imgs_paths
  /// \return
  bool set_images(
      const std::vector<std::string> &imgs_paths); // opt for future_batch

  /// \brief
  /// \return
  bool process_images();

  bool process_images(const std::vector<std::string> &imgs_paths);

  bool process_images(const std::vector<cv::Mat> &images);

  /// \brief Method for configuring wrapper by set all values explicitly
  /// \param input_size is a size for input images. Need to be set according to
  /// network architecture that are using.
  /// \param colors_path is a path to
  /// classes and according colors of dataset that network is pretrained on.
  /// \param pb_path is a path for pretrained protobuf file
  /// \param input_node is a name of input node
  /// \param output_node is a name of output node
  /// \return
  bool configure_wrapper(const cv::Size &input_size,
                         const std::string &colors_path,
                         const std::string &pb_path,
                         const std::string &input_node,
                         const std::string &output_node);

  bool prepare_for_inference(std::string config_path = "config.json");

  /// \brief
  /// \param resized
  /// \return
  std::vector<cv::Mat> get_indexed(bool resized); // =true

  /// \brief
  /// \param resized
  /// \return
  std::vector<cv::Mat> get_colored(bool resized); // =true

  /// \brief Method for getting all visible devices that can handle computations
  /// \return
  std::string get_devices();

protected:
  bool _is_configured = false;
  cv::Size _img_des_size;
  std::vector<cv::Size> _img_orig_size;
  std::vector<cv::Mat> _imgs;
  std::vector<cv::Mat> _result;
  std::unique_ptr<ISegmentationInterfaceHandler> inference_handler;
  std::unique_ptr<IDataBase> db_handler;
  std::vector<std::string> list_of_imgs;

  /// \brief Method for configuring wrapper if config need to be loaded from
  /// file
  /// \param config_path is a path to .json file with config to wrapper
  /// \return
  bool load_config(std::string config_path); // ="config.json"
};

#endif // TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H
