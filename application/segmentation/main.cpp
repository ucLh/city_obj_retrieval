#include "command_line_utils.h"
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tf_wrapper/segmentation_base.h>
#include <vector>

int main(int argc, char *argv[]) {

  std::string const in_file_name =
      cmd_utils::parse_command_line(argc, argv, std::string("-img"));
  std::string const is_colored =
      cmd_utils::parse_command_line(argc, argv, std::string("-colored"));

  std::vector<cv::Mat> output_indices;
  SegmentationWrapperBase seg_wrapper;

  seg_wrapper.load_config("config.json");

//  seg_wrapper.configure_wrapper(
//      cv::Size(1024, 1024),
//      "/home/luch/Programming/C++/city_obj_retrieval/classes.csv",
//      "/home/luch/Programming/C++/city_obj_retrieval/Xception-Deeplab.pb",
//      "ImageTensor:0", "SemanticPredictions:0");

  seg_wrapper.set_images({in_file_name});

  //    PROFILE_BLOCK("process images");
  if (!seg_wrapper.process_images())
    std::cerr << "Failed to process images" << std::endl;
  if ("true" == is_colored)
    output_indices = seg_wrapper.get_colored(true);
  else if ("false" == is_colored)
    output_indices = seg_wrapper.get_indices(true);
  else {
    std::cout << "Option not recognized" << std::endl;
    return 1;
  }
  for (unsigned long i = 0; i < output_indices.size(); ++i) {
    cv::imwrite(cv::format("out_%i.png", i), output_indices[i]);
  }

  std::cout << "Wrapper finished successfully" << std::endl;
  return 0;
}