#ifndef TF_WRAPPER_COMMON_OPS_H
#define TF_WRAPPER_COMMON_OPS_H

#include "fs_handling.h"

namespace common_ops {
/// \brief
/// \param filepath
/// \return
std::string extract_class(const std::string &filepath);

template <typename Type> inline void delete_safe(Type *&ptr) {
  delete ptr;
  // ptr = (Type *)(uintptr_t(NULL) - 1);        /* We are not hiding our
  // mistakes by zeroing the pointer */
  ptr = NULL;
}

template <typename Type> inline void deletearr_safe(Type *&ptr) {
  delete[] ptr;
  // ptr = (Type *)(uintptr_t(NULL) - 1);        /* We are not hiding our
  // mistakes by zeroing the pointer */
  ptr = NULL;
}

void check_end_slash(std::string &path);

std::vector<std::string> split(const std::string &text,
                               const std::string &delim);

std::string extract_class(const std::string &path_to_img,
                          std::string &series_path, std::string &queries_path);
} // namespace common_ops

#endif // TF_WRAPPER_COMMON_OPS_H
