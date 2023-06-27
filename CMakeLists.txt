cmake_minimum_required(VERSION 3.10)

project(ddrnet)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML REQUIRED yaml-cpp)
# tensorRT
set(TRT_PATH /opt/TensorRT)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${YAML_INCLUDE_DIRS}
    ${TRT_PATH}/include/
)
link_libraries(
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${YAML_LIBRARIES}
    ${TRT_PATH}/lib/libnvinfer.so
    ${TRT_PATH}/lib/libnvonnxparser.so
    ${TRT_PATH}/lib/libnvinfer_plugin.so
)
add_executable(app app.cc SegmentDetector.cpp ddrnet/DDRNet.cc)