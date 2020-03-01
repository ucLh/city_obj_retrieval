#include "tf_wrapper/common/common_ops.h"

std::string common_ops::extract_class(const std::string &filepath) {
  std::string spilt_delim = "/"; // TODO another delimiter for windows
  std::string token;
  std::string train_identifier = "series/";
  std::string test_identifier = "queries/";
  std::string classname_delim = "__";
  size_t pos_end;

  if (!filepath.empty()) {
    size_t pos_begin = filepath.find(test_identifier);
    if (std::string::npos == pos_begin) { // if not test directory
      pos_begin = filepath.find(train_identifier) +
                  train_identifier.size(); // find train directory identifier
      token = filepath.substr(pos_begin, std::string::npos);
      pos_end = token.find(spilt_delim);

      token = token.substr(0, pos_end);

      pos_end = token.find(classname_delim);
      if (std::string::npos != pos_end) {
        token = token.substr(0, pos_end);
      }

    } else { // we assume that we are in test directory
      pos_begin = filepath.find(test_identifier) + test_identifier.size();
      token = filepath.substr(pos_begin, std::string::npos);
      pos_end = token.find(spilt_delim);

      token = token.substr(0, pos_end);

      pos_end = token.find(classname_delim);
      if (std::string::npos != pos_end) {
        token = token.substr(0, pos_end);
      }
    }
  } else
    throw std::invalid_argument("Path is empty!");

  return token;
}
