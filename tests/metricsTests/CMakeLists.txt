set(MODULE_NAME metricsTests)
set(SRC_FILES
        ${CMAKE_CURRENT_LIST_DIR}/testMetricsBase.cpp
        )

add_executable(${MODULE_NAME} ${SRC_FILES} ${PROJECT_SOURCE_DIR}/application/metrics/metrics_base.cpp)


target_link_libraries(${MODULE_NAME} gtest gtest_main TF_WRAPPER_EMBEDDING)

target_include_directories(${MODULE_NAME} PUBLIC
        ${TF_WRAPPER_EMBEDDING_SOURCE_DIR})

target_include_directories(${MODULE_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/application/metrics)

#set(TEST_DATA_PATH ${CMAKE_BINARY_DIR}/tests/${MODULE_NAME})

add_test(
        NAME
        ${MODULE_NAME}
        COMMAND
        ./${MODULE_NAME}
)