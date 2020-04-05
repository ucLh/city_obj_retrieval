#include "command_line_utils.h"
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tf_wrapper/segmentation_base.h>
#include <vector>

int main(int argc, char *argv[]) {

  std::string const in_file_name =
      cmd_utils::parse_command_line(argc, argv, std::string("-img"));
  bool is_colored =
      cmd_utils::cmd_option_exists(argv, argv + argc, std::string("-colored"));

  std::vector<cv::Mat> output_indices;
  SegmentationWrapperBase seg_wrapper;

  seg_wrapper.prepare_for_inference("config.json");

  //    PROFILE_BLOCK("process images");
  if (!seg_wrapper.process_images())
    std::cerr << "Failed to process images" << std::endl;
  if (is_colored) {
    output_indices = seg_wrapper.get_colored(true);
  } else {
    output_indices = seg_wrapper.get_masked(true, {8, 13, 11});
  }
  for (unsigned long i = 0; i < output_indices.size(); ++i) {
    cv::imwrite(cv::format("out_%i.png", i), output_indices[i]);
  }

  std::cout << "Wrapper finished successfully" << std::endl;
  return 0;
}