# amalyshe - I added here version 2.8.12 just because it was used for DLDT, no other reasons so far
cmake_minimum_required(VERSION 2.8.12)
set (TARGET_NAME "tvm_hello_classification_ge")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(tvm_apps)
find_package( OpenCV REQUIRED )

set(TVM_HOME $ENV{TVM_HOME})
if(NOT DEFINED TVM_HOME)
        message(Error " Cannot find path to TVM, please set TVM_HOME environment variable")
else()
  message("Found TVM at ${TVM_HOME}")
endif()


add_executable(${TARGET_NAME} main.cpp)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OPENCV is disabled or not found, " ${TARGET_NAME} " build is interrupted")
else()
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES})
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENCV)
endif()

# add TVM headers to the lookup
include_directories("${TVM_HOME}/include"
                    "${TVM_HOME}/3rdparty/dmlc-core/include/"
                    "${TVM_HOME}/3rdparty/dlpack/include"
  )


target_compile_definitions(${TARGET_NAME} PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(${TARGET_NAME} PRIVATE "${TVM_HOME}/build/libtvm_runtime.dylib")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	target_link_libraries(${TARGET_NAME} PRIVATE "${TVM_HOME}/build/Release/tvm_runtime.lib")
else()
  target_link_libraries(${TARGET_NAME} PRIVATE "${TVM_HOME}/build/libtvm_runtime.so")
endif()