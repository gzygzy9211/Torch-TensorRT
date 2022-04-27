add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(torchtrt_cpp_header_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB_RECURSE torchtrt_cpp_headers RELATIVE ${torchtrt_cpp_header_dir}
     ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
message(STATUS "cpp_headers = ${torchtrt_cpp_headers}")
message(STATUS "cpp_header_dir = ${torchtrt_cpp_header_dir}")
