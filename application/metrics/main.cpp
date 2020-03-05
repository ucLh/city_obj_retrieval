#include "command_line_utils.h"
#include "metrics_base.h"
#include <iostream>

int main(int argc, char *argv[]) {
  std::string const inPath =
      cmd_utils::parse_command_line(argc, argv, std::string("--test_path"));
  std::string const topNClassesString =
      cmd_utils::parse_command_line(argc, argv, std::string("--topN"));
  int topNClasses = std::stoi(topNClassesString);
  std::cout << "Start initalizing tf_wrapper" << std::endl;
  MetricsBase tf_wrapper;
  std::cout << "Wrapper was initialized" << std::endl;
  tf_wrapper.getMetrics((std::string &)inPath, topNClasses);
}