#ifndef TF_WRAPPER_COMMAND_LINE_UTILS_H
#define TF_WRAPPER_COMMAND_LINE_UTILS_H

#include <algorithm>
#include <iostream>

namespace cmd_utils {

namespace {

char *get_cmd_option(char **begin, char **end, const std::string &option) {
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return nullptr;
}

bool cmd_option_exists(char **begin, char **end, const std::string &option) {
  return std::find(begin, end, option) != end;
}

} // namespace

std::string parse_command_line(int argc, char **argv, const std::string &c) {
  std::string ret;
  if (cmd_option_exists(argv, argv + argc, c)) {
    char *filename = get_cmd_option(argv, argv + argc, c);
    ret = std::string(filename);
  } else {
    std::cout << "Use -img $image$" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return ret;
}

} // namespace cmd_utils

#endif // TF_WRAPPER_COMMAND_LINE_UTILS_H
