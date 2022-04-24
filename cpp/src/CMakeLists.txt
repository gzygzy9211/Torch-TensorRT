add_library(cpp_interface OBJECT
    compile_spec.cpp
    logging.cpp
    ptq.cpp
    torch_tensorrt.cpp
    types.cpp
)
target_compile_definitions(cpp_interface
                           PRIVATE torch_tensorrt_EXPORT
                           PRIVATE torch_tensorrt_core_EXPORT)
target_include_directories(cpp_interface
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(cpp_interface nvinfer torch torch_library)
