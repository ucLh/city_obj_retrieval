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

std::vector<std::string> common_ops::split(const std::string &text,
                                           const std::string &delim) {
  std::vector<std::string> results = {};
  for (auto p = cbegin(text); p != cend(text);) {
    const auto n = search(p, cend(text), cbegin(delim), cend(delim));
    results.emplace_back(p, n);
    p = n;
    if (cend(text) != n) // we found delim, skip over it.
      p += delim.length();
  }
  return results;
}

std::string common_ops::check_end_slash(const std::string &path) {
  std::string result = path;
  if (result[result.size() - 1] == '/')
    result.pop_back();
  return result;
}

std::string common_ops::extract_class(const std::string &path_to_img,
                                       const std::string &series_path,
                                       const std::string &queries_path) {
  size_t class_pos = 0;
  std::string series_path_checked = check_end_slash(series_path);
  std::string queries_path_checked = check_end_slash(queries_path);
  auto series_dir = split(series_path_checked, "/").back();
  auto queries_dir = split(queries_path_checked, "/").back();

  size_t in_queries_pos = path_to_img.find(queries_dir);
  size_t in_imgs_pos = path_to_img.find(series_dir);

  if (in_queries_pos == std::string::npos)
    class_pos = in_imgs_pos + series_dir.size() + 1;
  else
    class_pos = in_queries_pos + queries_dir.size() + 1;

  std::string relative_img_path =
      path_to_img.substr(class_pos, std::string::npos);
  std::string classname = split(relative_img_path, "/")[0];
  classname = split(classname, "__")[0];
  return classname;
}