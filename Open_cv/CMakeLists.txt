cmake_minimum_required(VERSION 3.10)
project(OpenCV_Learning)

# --- 关键：手动指定 OpenCV 的路径 ---
set(OpenCV_DIR "E:/opencv/build") 

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(my_app 1.cpp)
target_link_libraries(my_app ${OpenCV_LIBS})