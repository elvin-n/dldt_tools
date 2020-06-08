
set (TARGET_NAME "snpe_backend")

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

include_directories(${SNPE_ROOT}/include/zdl)

add_library(${TARGET_NAME} SHARED ${MAIN_SRC} ${MAIN_HEADERS} )

if(NOT DEFINED ANDROID_NATIVE_API_LEVEL)
  target_link_libraries(${TARGET_NAME} PRIVATE ${SNPE_ROOT}/lib/x86_64-linux-clang/libSNPE.so)
else()
  target_link_libraries(${TARGET_NAME} PRIVATE ${SNPE_ROOT}/lib/arm-android-clang6.0/libSNPE.so)
endif()




