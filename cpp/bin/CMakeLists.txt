add_executable(torchtrtc torchtrtc/main.cpp)
target_link_libraries_recursive(torchtrtc cpp_interface core_compiler)
target_link_libraries(torchtrtc args nvinfer nvinfer_plugin torch torch_library)
target_compile_definitions(torchtrtc
                           PRIVATE torch_tensorrt_EXPORT
                           PRIVATE torch_tensorrt_core_EXPORT)