cmake_minimum_required(VERSION 3.6)

project(findPoint)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Where to find CMake modules and OpenCV
#set(OpenCV_DIR "D:\\ProgramData\\opencv\\cv2\\install")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(SRC_lib
        ${PROJECT_SOURCE_DIR}/circle.cpp
        )

add_executable(findPoint main.cpp stereoRectified.cpp)
target_link_libraries(findPoint ${OpenCV_LIBS} SRC_lib)
