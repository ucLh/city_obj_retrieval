#ifndef TF_WRAPPER_COMMON_OPS_H
#define TF_WRAPPER_COMMON_OPS_H

#include "fs_handling.h"

namespace common_ops {
/// \brief Function for extracting class (building name) from a filepath.
/// Requires 'series' or 'queries' string to be in a filepath
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

std::string check_end_slash(const std::string &path);

std::vector<std::string> split(const std::string &text,
                               const std::string &delim);

/// \brief Alternative version of class extracting function. Series path and
/// quires path now have to be set
/// \param path_to_img
/// \param series_path
/// \param queries_path
std::string extract_class(const std::string &path_to_img,
                           const std::string &series_path,
                           const std::string &queries_path);
} // namespace common_ops

#endif // TF_WRAPPER_COMMON_OPS_H
