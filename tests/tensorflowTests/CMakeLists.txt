set(MODULE_NAME tensorflowTests)
set(SRC_FILES
        ${CMAKE_CURRENT_LIST_DIR}/testExecutionAcceptance.cpp
        )

add_executable(${MODULE_NAME} ${SRC_FILES})

set(TEST_DATA_PATH_QUERIES ${CMAKE_BINARY_DIR}/tests/${MODULE_NAME}/queries/helsinki_1_andante)

set(TEST_DATA_PATH_SERIES ${CMAKE_BINARY_DIR}/tests/${MODULE_NAME}/series)

file(GLOB_RECURSE TEST_DATA_1 ${CMAKE_CURRENT_LIST_DIR}/../test_data/series/helsinki_1_andante/images/*.jpg)
file(GLOB_RECURSE TEST_DATA_2 ${CMAKE_CURRENT_LIST_DIR}/../test_data/series/helsinki_2_apteeki/images/*.jpg)
file(GLOB_RECURSE TEST_DATA_3 ${CMAKE_CURRENT_LIST_DIR}/../test_data/series/helsinki_3_usadba/images/*.jpg)


file(COPY ${TEST_DATA_1} DESTINATION ${TEST_DATA_PATH_SERIES}/helsinki_1_andante/images/)
file(COPY ${TEST_DATA_2} DESTINATION ${TEST_DATA_PATH_SERIES}/helsinki_2_apteeki/images/)
file(COPY ${TEST_DATA_2} DESTINATION ${TEST_DATA_PATH_SERIES}/helsinki_3_usadba/images/)

file(MAKE_DIRECTORY ${TEST_DATA_PATH_QUERIES})

file(COPY ${CMAKE_CURRENT_LIST_DIR}/../test_data/queries/helsinki_1_andante/IMG_7813_4.jpg DESTINATION ${TEST_DATA_PATH_QUERIES})

target_link_libraries(${MODULE_NAME} gtest gtest_main TF_WRAPPER_EMBEDDING)

target_include_directories(${MODULE_NAME} PUBLIC
        ${TF_WRAPPER_EMBEDDING_SOURCE_DIR})

add_test(
        NAME
        ${MODULE_NAME}
        COMMAND
        ./${MODULE_NAME}
)
