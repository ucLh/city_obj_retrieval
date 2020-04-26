set(MODULE_NAME embeddingsTests)
set(SRC_FILES
        ${CMAKE_CURRENT_LIST_DIR}/testEmbeddingsBase.cpp
        #        ${CMAKE_CURRENT_LIST_DIR}/testTensorflowCorrectness.cpp
        )

add_executable(${MODULE_NAME} ${SRC_FILES})


target_link_libraries(${MODULE_NAME} gtest gtest_main TF_WRAPPER_EMBEDDING)

target_include_directories(${MODULE_NAME} PUBLIC
        ${TF_WRAPPER_EMBEDDING_SOURCE_DIR})

set(TEST_DATA_PATH ${CMAKE_BINARY_DIR}/tests/${MODULE_NAME})
file(COPY ${CMAKE_CURRENT_LIST_DIR}/../test_data/sample_images/Lenna.jpg DESTINATION ${TEST_DATA_PATH})
file(COPY ${CMAKE_CURRENT_LIST_DIR}/../test_data/config.json DESTINATION ${TEST_DATA_PATH})
file(COPY ${CMAKE_CURRENT_LIST_DIR}/../test_data/testdatafile.txt DESTINATION ${TEST_DATA_PATH})

add_test(
        NAME
        ${MODULE_NAME}
        COMMAND
        ./${MODULE_NAME}
)