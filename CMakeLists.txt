cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Torch-TensorRT CXX CUDA)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils.cmake)

set(CMAKE_BUILD_TYPE Release)
if(MSVC)
    set(CMAKE_CONFIGURATION_TYPES Release)
endif()

set(CMAKE_CXX_STANDARD 14)

include(${CMAKE_CURRENT_SOURCE_DIR}/third_party/third_party.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpp)
foreach(var torchtrt_core_headers torchtrt_core_header_dir)
    get_directory_property(${var} DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/core DEFINITION ${var})
endforeach()
foreach(var torchtrt_cpp_headers torchtrt_cpp_header_dir)
    get_directory_property(${var} DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cpp DEFINITION ${var})
endforeach()


set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/py/torch_tensorrt)
install(TARGETS torchtrtc DESTINATION "bin")
install(TARGETS torchtrt DESTINATION "lib")
install(TARGETS torchtrt_runtime DESTINATION "lib")
install(TARGETS torchtrt_plugins DESTINATION "lib")
install_headers(${torchtrt_cpp_headers} ${torchtrt_cpp_header_dir} include)
install_headers(${torchtrt_core_headers} ${torchtrt_core_header_dir} include/torch_tensorrt)
