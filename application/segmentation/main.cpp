#include "command_line_utils.h"
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tf_wrapper/segmentation_base.h>
#include <vector>

int main(int argc, char *argv[]) {

  std::string const in_file_name =
      cmd_utils::parse_command_line(argc, argv, std::string("-img"));

  std::vector<cv::Mat> output_indexed, output_colored, output_masked;
  SegmentationWrapper seg_wrapper;

  seg_wrapper.prepare_for_inference("config.json");

  if (!seg_wrapper.process_images({in_file_name})) {
    std::cerr << "Failed to process images" << std::endl;
  }
  output_indexed = seg_wrapper.get_indexed(true);
  output_colored = seg_wrapper.get_colored(true);
  // Here we are masking trees, pedestrians and cars
  output_masked = seg_wrapper.get_masked(true, {8, 11, 13});

  cv::imwrite("indexed.png", output_indexed[0]);
  cv::imwrite("colored.png", output_colored[0]);
  cv::imwrite("masked.png", output_masked[0]);

  std::cout << "Wrapper finished successfully" << std::endl;
  return 0;
}