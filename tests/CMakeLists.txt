MESSAGE(STATUS "Including GTest on Tests build")
#include_directories (${GTest_INCLUDE_DIR}/..)

set(CMAKE_MODULE_PATH
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_MODULE_PATH}../cmake)


find_package(TensorFlow CONFIG REQUIRED)

# Configuring test data

file(COPY test_data/val80_optimized.pb DESTINATION ${CMAKE_BINARY_DIR}/test_data/)
file(GLOB_RECURSE TEST_DATA ${CMAKE_CURRENT_LIST_DIR}/test_data/series/*.jpg)

#message(STATUS ${TEST_DATA})
#message(STATUS ${TEST_DATA_DIRECTORY})


add_subdirectory(commonTests)
add_subdirectory(tensorflowTests)
add_subdirectory(wrapperTests)
add_subdirectory(metricsTests)

configure_file(test_data/config.json ${CMAKE_BINARY_DIR}/tests/tensorflowTests/config.json COPYONLY)
configure_file(test_data/sample_config.json ${CMAKE_BINARY_DIR}/tests/commonTests/sample_config.json COPYONLY)
