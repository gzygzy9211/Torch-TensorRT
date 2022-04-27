add_library(core_common INTERFACE IMPORTED)
target_include_directories(core_common 
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/.."
    INTERFACE "\$<TARGET_PROPERTY:nvinfer,INTERFACE_INCLUDE_DIRECTORIES>"
    INTERFACE "\$<TARGET_PROPERTY:torch,INTERFACE_INCLUDE_DIRECTORIES>"
    INTERFACE "\$<TARGET_PROPERTY:torch_library,INTERFACE_INCLUDE_DIRECTORIES>"
)
target_compile_definitions(core_common INTERFACE torch_tensorrt_core_EXPORT)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/util)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/plugins)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/runtime)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/conversion)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lowering)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/partitioning)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ir)

add_library(core_compiler OBJECT
            compiler.cpp)
target_link_libraries(core_compiler
            PUBLIC core_conversion
            PUBLIC core_runtime
            PUBLIC core_lowering
            PUBLIC core_partitioning
            PUBLIC core_util
            PUBLIC core_ir
            PUBLIC core_plugins
            PRIVATE core_common)

set(torchtrt_core_header_dir ${CMAKE_CURRENT_SOURCE_DIR}/..)
file(GLOB_RECURSE torchtrt_core_headers RELATIVE ${torchtrt_core_header_dir}
     ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
list(REMOVE_ITEM torchtrt_core_headers
     "core/conversion/evaluators/eval_macros.h"
     "core/conversion/evaluators/eval_util.h")
message(STATUS "core_headers = ${torchtrt_core_headers}")
message(STATUS "core_header_dir = ${torchtrt_core_header_dir}")
