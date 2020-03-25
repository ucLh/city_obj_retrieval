cmake_minimum_required(VERSION 3.10)
project(TF_WRAPPER_EMBEDDING)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")
#set(CMAKE_CXX_FLAGS "-lpthread")
#set(CMAKE_CXX_FLAGS_DEBUG "-lpthread")
#set(CMAKE_CXX_FLAGS_RELEASE "-lpthread")

find_package(TensorFlow CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(Protobuf REQUIRED)

add_library(TF_WRAPPER_EMBEDDING src/tensorflow_base.cpp
        src/tensorflow_wrapper_core.cpp
        src/tensorflow_embeddings.cpp
        src/tensorflow_auxiliary.cpp
        src/embeddings_base.cpp
        src/wrapper_legacy.cpp
        src/segmentation_base.cpp
        src/tensorflow_segmentator.cpp
        src/common/fs_handling.cpp
        src/common/common_ops.cpp
        )

target_include_directories(TF_WRAPPER_EMBEDDING PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(TF_WRAPPER_EMBEDDING PRIVATE ${TensorFlow_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(TF_WRAPPER_EMBEDDING PRIVATE stdc++fs ${OpenCV_LIBS} ${TensorFlow_LIBRARIES} pthread)

message(TF LIBRARIES ${TensorFlow_LIBRARIES})

message("INCLUDING TF LIBS" ${TensorFlow_LIBRARIES})
#message("${TensorFlow_INCLUDE_DIRS}")
