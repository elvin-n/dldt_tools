# amalyshe - I added here version 2.8.12 just because it was used for DLDT, no other reasons so far
cmake_minimum_required(VERSION 2.8.12)

project(tflite_apps)
find_package( OpenCV REQUIRED )

add_executable(tflite_hello_classification main.cpp)

if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OPENCV is disabled or not found, " ${TARGET_NAME} " build is interrupted")
else()
    target_link_libraries(tflite_hello_classification PRIVATE ${OpenCV_LIBRARIES})
  target_compile_definitions(tflite_hello_classification PRIVATE USE_OPENCV)
endif()

# add SNPE headers to the lookup
include_directories(${TENSORFLOW_ROOT})
include_directories(${TENSORFLOW_ROOT}/tensorflow/lite/tools/make/downloads/flatbuffers/include)

if(NOT DEFINED ANDROID_NATIVE_API_LEVEL)
  target_link_libraries(tflite_hello_classification PRIVATE ${TENSORFLOW_ROOT}/bazel-out/k8-opt/bin/tensorflow/lite/libtensorflowlite.so)
else()
  target_link_libraries(tflite_hello_classification PRIVATE ${TENSORFLOW_ROOT}/bazel-out/arm64-v8a-opt/bin/tensorflow/lite/libtensorflowlite.so
          ${TENSORFLOW_ROOT}/bazel-out/arm64-v8a-opt/bin/external/flatbuffers/src/libflatbuffers.a)
endif()


# target_link_libraries(snpe_hello_classification $(OpenCV_LIBS))
#                            OPENCV_DEPENDENCIES imgcodecs)
