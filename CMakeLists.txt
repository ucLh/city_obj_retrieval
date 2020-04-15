cmake_minimum_required(VERSION 3.10)
project(TF_WRAPPER)
set(CMAKE_PREFIX_PATH ~/.local)

set(CMAKE_CXX_STANDARD 17)
set(PROJECT_NAME TF_WRAPPER)

add_subdirectory(tf_wrapper)
add_subdirectory(application/embeddings)
add_subdirectory(application/metrics)
add_subdirectory(application/segmentation)

include(cmake/googletest.cmake)
fetch_googletest(
        ${PROJECT_SOURCE_DIR}/cmake
        ${PROJECT_BINARY_DIR}/googletest
)

enable_testing()
add_subdirectory(tests)