
set (TARGET_NAME "dldt_backend")

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

add_library(${TARGET_NAME} SHARED ${MAIN_SRC} ${MAIN_HEADERS} )
target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES})

