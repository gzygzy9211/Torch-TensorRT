add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/passes)
add_library(core_lowering OBJECT
    drop_unused_nodes.cpp
    lowering.cpp
    register_trt_placeholder_ops.cpp
    LowerInfo.cpp
)
target_link_libraries(core_lowering PUBLIC core_util PUBLIC core_lowering_passes PRIVATE core_common)
