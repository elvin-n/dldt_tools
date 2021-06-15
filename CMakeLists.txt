# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 2.8.12)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(DLDT_Tools DESCRIPTION "Tools for different inference engines")

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT(BIN_FOLDER))
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l")
        set (ARCH armv7l)
    elseif("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        set (ARCH intel64)
    else()
        set (ARCH ia32)
    endif()

    set (BIN_FOLDER ${ARCH})
endif()

if (NOT(IE_MAIN_SOURCE_DIR))
    # in case if samples are built out of IE repo
    set (IE_MAIN_SAMPLES_DIR ${CMAKE_CURRENT_BINARY_DIR})
else()
    # in case if samples are built from IE repo
    set (IE_MAIN_SAMPLES_DIR ${IE_MAIN_SOURCE_DIR})
endif()

if(NOT(UNIX))
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_LIBRARY_PATH ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}) # compatibility issue: linux uses LIBRARY_OUTPUT_PATH, windows uses LIBRARY_OUTPUT_DIRECTORY
else ()
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}/lib)
endif()

# use this flag if you need to throw custom message in case if the IE package is not found.
find_package(InferenceEngine 2.1 QUIET)
if (NOT DEFINED InferenceEngine_FOUND)
    message(WARNING " Not found Intel OpenVINO, will ignore this flow")
else()
    message("Found Intel OpenVINO at ${InferenceEngine_INCLUDE_DIRS}")
endif()

set(SNPE_ROOT $ENV{SNPE_ROOT})
if(NOT DEFINED SNPE_ROOT)
	message(Warning " SNPE cannot be found, please set SNPE_ROT environemnt variables to compile SNPE dedicated components")
else()
  message("Found SNPE at ${SNPE_ROOT}")
endif()

set(TENSORFLOW_ROOT $ENV{TENSORFLOW_ROOT})
if(NOT DEFINED TENSORFLOW_ROOT)
	message(Warning " TensorFlow cannot be found, please set TENSORFLOW_ROOT env variable to compile Tensorflow Lite dedicated components")
else()
  message("Found TensorFlow at ${TENSORFLOW_ROOT}")
endif()

set(TVM_HOME $ENV{TVM_HOME})
if(NOT DEFINED TVM_HOME)
  message(Warning " TVM cannot be found, please set TVM_HOME env variable to compile TVM dedicated components")
else()
  message("Found TVM at ${TVM_HOME}")
endif()


if (WIN32)
    if (NOT "${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        message(FATAL_ERROR "Only 64-bit supported on Windows")
    endif()

    set_property (DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")
else()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Werror=return-type ")
    if (APPLE)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument")
    elseif(UNIX)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self")
        if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
        endif()
    endif()
endif()


####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
####################################

set (GFLAGS_IS_SUBPROJECT TRUE)
set (HAVE_SYS_STAT_H 1)
set (HAVE_INTTYPES_H 1)

if (WIN32)
    # add_compile_options("/WX")
else()
    add_compile_options("-Werror")
endif()

# Properties->C/C++->General->Additional Include Directories
include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/common/format_reader
    ${InferenceEngine_INCLUDE_DIRS}
    ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/gflags/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

if (UNIX)
    set (LIB_DL dl)
endif()

add_subdirectory(thirdparty/gflags)
add_subdirectory(thirdparty/gtest)
add_subdirectory(common/format_reader)

# collect all samples subdirectories
file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
# skip building of unnecessary subdirs
list(REMOVE_ITEM subdirs archived common thirdparty)

add_subdirectory(validation_app)
if (DEFINED InferenceEngine_FOUND)
  if(NOT DEFINED ANDROID_NATIVE_API_LEVEL)
    # add_subdirectory(calibration_tool)
    add_subdirectory(ie_backend)
  endif()
endif()

if(DEFINED SNPE_ROOT)
  add_subdirectory(snpe_samples/snpe_hello_classification)
  add_subdirectory(snpe_samples/snpe_object_detection_ssd)
  add_subdirectory(snpe_backend)
endif()

if(DEFINED TENSORFLOW_ROOT)
  add_subdirectory(tflite_samples/tflite_hello_classification)
  add_subdirectory(tflite_samples/tflite_object_detection_ssd)
  add_subdirectory(tflite_backend)
endif()

if(DEFINED TVM_HOME)
  add_subdirectory(tvm_backend)
endif()
