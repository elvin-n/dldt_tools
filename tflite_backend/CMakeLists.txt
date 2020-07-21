
set (TARGET_NAME "tflite_backend")

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        )

source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/../validation_app
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

include_directories(${TENSORFLOW_ROOT})
include_directories(${TENSORFLOW_ROOT}/tensorflow/lite/tools/make/downloads/flatbuffers/include)


add_library(${TARGET_NAME} SHARED ${MAIN_SRC} ${MAIN_HEADERS} )

if(NOT DEFINED ANDROID_NATIVE_API_LEVEL)
  target_link_libraries(${TARGET_NAME} PRIVATE ${TENSORFLOW_ROOT}/bazel-out/k8-opt/bin/tensorflow/lite/libtensorflowlite.so)
else()
  if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_link_libraries(${TARGET_NAME} PRIVATE ${TENSORFLOW_ROOT}/bazel-out/arm64-v8a-opt/bin/tensorflow/lite/libtensorflowlite.so)
  else()
    target_link_libraries(${TARGET_NAME} PRIVATE
            ${TENSORFLOW_ROOT}/bazel-out/armeabi-v7a-opt/bin/tensorflow/lite/libtensorflowlite.so
            ${TENSORFLOW_ROOT}/bazel-out/armeabi-v7a-opt/bin/tensorflow/lite/delegates/hexagon/libhexagon_delegate.a
            ${TENSORFLOW_ROOT}/bazel-out/armeabi-v7a-opt/bin/tensorflow/lite/delegates/hexagon/libhexagon_implementation.a
            ${TENSORFLOW_ROOT}/bazel-out/armeabi-v7a-opt/bin/tensorflow/lite/delegates/hexagon/libhexagon_delegate_kernel.a
            ${TENSORFLOW_ROOT}/bazel-out/armeabi-v7a-opt/bin/tensorflow/lite/delegates/hexagon/libutils.a
            ${TENSORFLOW_ROOT}/bazel-out/armeabi-v7a-opt/bin/tensorflow/lite/delegates/hexagon/builders/libop_builder.a
    )
  endif()
endif()




