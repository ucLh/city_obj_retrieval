#include "command_line_utils.h"
#include "tf_wrapper/wrapper_base.h"
#include <iostream>

#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  std::string const in_file_name =
      cmd_utils::parse_command_line(argc, argv, std::string("-img"));

  WrapperBase tf_wrapper;
  if (!tf_wrapper.prepare_for_inference()) {
    std::cerr << "Can't prepare for inference!" << std::endl;
    return 1;
  }
  //    tf_wrapper->topN = 10;

  std::vector<WrapperBase::distance> results =
      tf_wrapper.inference_and_matching(in_file_name);
  //    common_ops::delete_safe(tf_wrapper);

  for (const auto &result : results)
    std::cout << "Dst " << result.dist << " path " << result.path << std::endl;

  return 0;
}