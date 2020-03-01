#include "metrics_base.h"
#include <iostream>

char *getCmdOption(char **begin, char **end, const std::string &option) {
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return nullptr;
}

bool cmdOptionExists(char **begin, char **end, const std::string &option) {
  return std::find(begin, end, option) != end;
}

std::string parseCommandLine(int argc, char *argv[], const std::string &c) {
  std::string ret;
  if (cmdOptionExists(argv, argv + argc, c)) {
    char *filename = getCmdOption(argv, argv + argc, c);
    ret = std::string(filename);
  } else {
    std::cout << "Use --test_path $path for images to test$" << std::endl;
    std::cout << "Use --topN $number of classes$" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return ret;
}

int main(int argc, char *argv[]) {
  std::string const inPath =
      parseCommandLine(argc, argv, std::string("--test_path"));
  std::string const topNClassesString =
      parseCommandLine(argc, argv, std::string("--topN"));
  int topNClasses = std::stoi(topNClassesString);
  std::cout << "Start initalizing tf_wrapper" << std::endl;
  MetricsBase tf_wrapper;
  std::cout << "Wrapper was initialized" << std::endl;
  tf_wrapper.getMetrics((std::string &)inPath, topNClasses);
}