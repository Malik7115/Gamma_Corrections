cmake_minimum_required(VERSION 3.0.0)
project(Inference_cpp VERSION 0.1.0)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(Inference_cpp main.cpp)
target_link_libraries(Inference_cpp ${OpenCV_LIBS} /home/ibrahim/Softwares/onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
