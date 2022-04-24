add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/bin)

file(GLOB_RECURSE torchtrt_cpp_headers ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
