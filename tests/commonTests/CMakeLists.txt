set(MODULE_NAME commonTests)
set(SRC_FILES
        ${CMAKE_CURRENT_LIST_DIR}/testClassExtraction.cpp
        ${CMAKE_CURRENT_LIST_DIR}/testDataHandling.cpp
        )

add_executable(${MODULE_NAME} ${SRC_FILES})

set(TEST_DATAFILE_PATH ${CMAKE_BINARY_DIR}/tests/${MODULE_NAME}/this/is/test/path)
set(TEST_DATA_PATH ${CMAKE_BINARY_DIR}/tests/${MODULE_NAME})

file(MAKE_DIRECTORY ${TEST_DATAFILE_PATH})

target_link_libraries(${MODULE_NAME} gtest gtest_main TF_WRAPPER_EMBEDDING)

configure_file(../test_data/testdatafile.txt ${TEST_DATAFILE_PATH} COPYONLY)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/../test_data/sample_images/Lenna.jpg DESTINATION ${TEST_DATA_PATH})

target_include_directories(${MODULE_NAME} PUBLIC
        ${TF_WRAPPER_EMBEDDING_SOURCE_DIR})

add_test(
        NAME
        ${MODULE_NAME}
        COMMAND
        ./${MODULE_NAME}
)