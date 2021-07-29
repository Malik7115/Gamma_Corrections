cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch_inference)

find_package(Torch REQUIRED)  # find libtorch
find_package(OpenCV REQUIRED)  # find OpenCV

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(libtorch_inference main.cpp)
target_link_libraries(libtorch_inference ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET libtorch_inference PROPERTY CXX_STANDARD 14)