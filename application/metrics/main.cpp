#include "command_line_utils.h"
#include "metrics_base.h"
#include <iostream>

int main(int argc, char *argv[]) {
  std::string const in_path =
      cmd_utils::parse_command_line(argc, argv, std::string("--test_path"));
  std::string const top_n_classesString =
      cmd_utils::parse_command_line(argc, argv, std::string("--top_n_classes"));
  bool use_segmentation = cmd_utils::cmd_option_exists(
      argv, argv + argc, std::string("--use_segmentation"));
  int top_n_classes = std::stoi(top_n_classesString);
  std::cout << "Start initalizing tf_wrapper" << std::endl;
  MetricsBase tf_wrapper;
  std::cout << "Wrapper was initialized" << std::endl;
  tf_wrapper.get_metrics((std::string &)in_path, top_n_classes,
                         use_segmentation);
}