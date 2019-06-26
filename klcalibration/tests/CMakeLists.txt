# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

set(TARGET_NAME "CalibrationUnitTests")

#rpath enabled for unit tests only
SET (CMAKE_SKIP_RPATH OFF)

# collect sources

file(GLOB MAIN_SRC
        *.cpp
	../data_stats.cpp
        )


include_directories("../"
        )

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC})

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE" 
	COMPILE_PDB_NAME ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE
    gtest
    gtest_main
    gmock
    gflags)

add_test(NAME ${TARGET_NAME}
        COMMAND ${TARGET_NAME})
