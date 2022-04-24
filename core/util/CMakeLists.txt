add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/logging)

add_library(core_util OBJECT
    Exception.cpp
    trt_util.cpp
)
target_link_libraries(core_util PUBLIC core_util_logging PRIVATE core_common)
