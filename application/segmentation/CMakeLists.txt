project(TF_WRAPPER_SEGMENTATION_EXAMPLE)

find_package(OpenCV REQUIRED)

add_executable(TF_SEGMENTATION_EXAMPLE main.cpp ../utils/command_line_utils.h)

configure_file(config.json ${CMAKE_BINARY_DIR}/application/segmentation COPYONLY)

target_link_libraries(TF_SEGMENTATION_EXAMPLE TF_WRAPPER_EMBEDDING ${OpenCV_LIBS})
target_include_directories(TF_SEGMENTATION_EXAMPLE
        PUBLIC
        ${TF_WRAPPER_SEGMENTATION_SOURCE_DIR}
        ../utils
        )