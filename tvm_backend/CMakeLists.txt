set (TARGET_NAME "tvm_backend")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

include_directories("${TVM_HOME}/include"
                    "${TVM_HOME}/3rdparty/dmlc-core/include/"
                    "${TVM_HOME}/3rdparty/dlpack/include"
  )

add_library(${TARGET_NAME} SHARED ${MAIN_SRC} ${MAIN_HEADERS} )
target_compile_definitions(${TARGET_NAME} PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(${TARGET_NAME} PRIVATE "${TVM_HOME}/build/libtvm_runtime.dylib")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	target_link_libraries(${TARGET_NAME} PRIVATE "${TVM_HOME}/build/Release/tvm_runtime.lib")
else()
  target_link_libraries(${TARGET_NAME} PRIVATE "${TVM_HOME}/build/libtvm_runtime.so")
endif()

