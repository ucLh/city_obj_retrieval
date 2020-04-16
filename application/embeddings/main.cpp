#include "command_line_utils.h"
#include "tf_wrapper/embeddings_wrapper.h"
#include <iostream>

#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  std::string const in_file_name =
      cmd_utils::parse_command_line(argc, argv, std::string("-img"));

  EmbeddingsWrapper tf_wrapper;
  if (!tf_wrapper.prepare_for_inference("config.json")) {
    std::cerr << "Can't prepare for inference!" << std::endl;
    return 1;
  }

  std::vector<EmbeddingsWrapper::distance> results =
      tf_wrapper.inference_and_matching(in_file_name);
  results = tf_wrapper.inference_and_matching(in_file_name);

  for (const auto &result : results)
    std::cout << "Dst " << result.dist << " path " << result.path << std::endl;

  return 0;
}