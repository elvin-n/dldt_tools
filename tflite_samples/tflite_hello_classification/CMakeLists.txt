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

# add TFLite headers to the lookup
include_directories(${TENSORFLOW_ROOT})
include_directories(${TENSORFLOW_ROOT}/tensorflow/lite/tools/make/downloads/flatbuffers/include)

if(NOT DEFINED ANDROID_NATIVE_API_LEVEL)
  target_link_libraries(tflite_hello_classification PRIVATE ${TENSORFLOW_ROOT}/bazel-out/k8-opt/bin/tensorflow/lite/libtensorflowlite.so)
else()
  if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_link_libraries(tflite_hello_classification PRIVATE 
      ${TENSORFLOW_ROOT}/bazel-out/arm64-v8a-opt/bin/tensorflow/lite/libtensorflowlite.so
      ${TENSORFLOW_ROOT}/bazel-out/arm64-v8a-opt/bin/tensorflow/lite/delegates/hexagon/libhexagon_delegate.a
      ${TENSORFLOW_ROOT}/bazel-out/arm64-v8a-opt/bin/tensorflow/lite/delegates/hexagon/libhexagon_implementation.a
      ${TENSORFLOW_ROOT}/bazel-out/arm64-v8a-opt/bin/tensorflow/lite/delegates/hexagon/libhexagon_delegate_kernel.a
      ${TENSORFLOW_ROOT}/bazel-out/arm64-v8a-opt/bin/tensorflow/lite/delegates/hexagon/libutils.a
      ${TENSORFLOW_ROOT}/bazel-out/arm64-v8a-opt/bin/tensorflow/lite/delegates/utils/libsimple_delegate.a
      ${TENSORFLOW_ROOT}/bazel-out/arm64-v8a-opt/bin/tensorflow/lite/delegates/libutils.a
      ${TENSORFLOW_ROOT}/bazel-out/arm64-v8a-opt/bin/tensorflow/lite/delegates/hexagon/builders/libop_builder.a
      ${TENSORFLOW_ROOT}/bazel-out/arm64-v8a-opt/bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so
      )
  else()
    target_link_libraries(tflite_hello_classification PRIVATE
            ${TENSORFLOW_ROOT}/bazel-out/armeabi-v7a-opt/bin/tensorflow/lite/libtensorflowlite.so
            ${TENSORFLOW_ROOT}/bazel-out/armeabi-v7a-opt/bin/tensorflow/lite/delegates/hexagon/libhexagon_delegate.a
            ${TENSORFLOW_ROOT}/bazel-out/armeabi-v7a-opt/bin/tensorflow/lite/delegates/hexagon/libhexagon_implementation.a
            ${TENSORFLOW_ROOT}/bazel-out/armeabi-v7a-opt/bin/tensorflow/lite/delegates/hexagon/libhexagon_delegate_kernel.a
            ${TENSORFLOW_ROOT}/bazel-out/armeabi-v7a-opt/bin/tensorflow/lite/delegates/hexagon/libutils.a
            ${TENSORFLOW_ROOT}/bazel-out/armeabi-v7a-opt/bin/tensorflow/lite/delegates/hexagon/builders/libop_builder.a
    )
  endif()
endif()