project(TF_WRAPPER_EMBEDDING_EXAMPLE)

add_executable(TF_EMBEDDINGS_EXAMPLE main.cpp ../utils/command_line_utils.h)

target_link_libraries(TF_EMBEDDINGS_EXAMPLE TF_WRAPPER_EMBEDDING)
message(dir ${TF_WRAPPER_EMBEDDING_SOURCE_DIR})

configure_file(../../config.json ${CMAKE_BINARY_DIR}/application/embeddings COPYONLY)
target_include_directories(TF_EMBEDDINGS_EXAMPLE
        PUBLIC
        ${TF_WRAPPER_EMBEDDING_SOURCE_DIR}
        ../utils
        )
