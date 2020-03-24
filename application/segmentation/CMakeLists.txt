project(TF_WRAPPER_SEGMENTATION_EXAMPLE)

find_package(OpenCV REQUIRED)

add_executable(TF_SEGMENTATOR main.cpp ../utils/command_line_utils.h)

configure_file(config.json ${CMAKE_BINARY_DIR}/application/segmentation COPYONLY)

target_link_libraries(TF_SEGMENTATOR TF_WRAPPER_EMBEDDING ${OpenCV_LIBS})
target_include_directories(TF_SEGMENTATOR
        PUBLIC
        ${TF_WRAPPER_SEGMENTATION_SOURCE_DIR}
        ../utils
        )