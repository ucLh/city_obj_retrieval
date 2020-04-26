set(MODULE_NAME segmentationTests)
set(SRC_FILES
        ${CMAKE_CURRENT_LIST_DIR}/testSegmentationBase.cpp
        )

add_executable(${MODULE_NAME} ${SRC_FILES})


target_link_libraries(${MODULE_NAME} gtest gtest_main TF_WRAPPER_EMBEDDING)

target_include_directories(${MODULE_NAME} PUBLIC ${TF_WRAPPER_EMBEDDING_SOURCE_DIR})

set(TEST_DATA_PATH ${CMAKE_BINARY_DIR}/tests/${MODULE_NAME})
file(COPY ${CMAKE_CURRENT_LIST_DIR}/../test_data/segmented_images DESTINATION ${TEST_DATA_PATH})
file(COPY ${CMAKE_CURRENT_LIST_DIR}/../test_data/queries/helsinki_2_apteeki DESTINATION ${TEST_DATA_PATH})
file(COPY ${CMAKE_CURRENT_LIST_DIR}/../test_data/series/helsinki_3_usadba DESTINATION ${TEST_DATA_PATH})
file(COPY ${CMAKE_CURRENT_LIST_DIR}/../test_data/seg_config.json DESTINATION ${TEST_DATA_PATH})

add_test(NAME ${MODULE_NAME} COMMAND ./${MODULE_NAME})