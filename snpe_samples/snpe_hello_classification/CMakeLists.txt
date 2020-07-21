# amalyshe - I added here version 2.8.12 just because it was used for DLDT, no other reasons so far
cmake_minimum_required(VERSION 2.8.12)

# project(snpe_apps)
find_package( OpenCV REQUIRED )

add_executable(snpe_hello_classification main.cpp)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OPENCV is disabled or not found, " ${TARGET_NAME} " build is interrupted")
else()
    target_link_libraries(snpe_hello_classification PRIVATE ${OpenCV_LIBRARIES})
    target_compile_definitions(snpe_hello_classification PRIVATE USE_OPENCV)
endif()

# add SNPE headers to the lookup
set(SNPE_ROOT $ENV{SNPE_ROOT})
if(NOT DEFINED SNPE_ROOT)
  message(FATAL_ERROR " SNPE cannot be found, please set SNPE_ROOT env variable")
else()
  message("Found SNPE at ${SNPE_ROOT}")
endif()

include_directories(${SNPE_ROOT}/include/zdl)

if(NOT DEFINED ANDROID_NATIVE_API_LEVEL)
  target_link_libraries(snpe_hello_classification PRIVATE ${SNPE_ROOT}/lib/x86_64-linux-clang/libSNPE.so)
else()
  if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_link_libraries(snpe_hello_classification PRIVATE ${SNPE_ROOT}/lib/aarch64-android-clang6.0/libSNPE.so)
  else()
    target_link_libraries(snpe_hello_classification PRIVATE ${SNPE_ROOT}/lib/arm-android-clang6.0/libSNPE.so)
  endif()
endif()


# target_link_libraries(snpe_hello_classification $(OpenCV_LIBS))
#                            OPENCV_DEPENDENCIES imgcodecs)
