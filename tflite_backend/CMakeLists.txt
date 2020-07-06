
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
  target_link_libraries(${TARGET_NAME} PRIVATE ${SNPE_ROOT}/lib/arm-android-clang6.0/libSNPE.so)
endif()




