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

# set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/py/torch_tensorrt)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install/torch_tensorrt)
install(TARGETS torchtrtc DESTINATION "bin")
install(TARGETS torchtrt DESTINATION "lib")
install(TARGETS torchtrt_runtime DESTINATION "lib")
install(TARGETS torchtrt_plugins DESTINATION "lib")
install(FILES ${torchtrt_cpp_headers} DESTINATION "include")
