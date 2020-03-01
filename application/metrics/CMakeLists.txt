project(TF_WRAPPER_EMBEDDING_METRICS)

add_executable(TF_WRAPPER_METRICS main.cpp metrics_base.cpp metrics_base.h)

target_link_libraries(TF_WRAPPER_METRICS TF_WRAPPER_EMBEDDING)
message(dir ${TF_WRAPPER_EMBEDDING_SOURCE_DIR})

configure_file(config.json ${CMAKE_BINARY_DIR}/application/metrics COPYONLY)

target_include_directories(TF_WRAPPER_METRICS
        PUBLIC
        ${TF_WRAPPER_EMBEDDING_SOURCE_DIR}
        )
