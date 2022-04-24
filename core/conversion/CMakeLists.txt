
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/var)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tensorcontainer)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/conversionctx)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/converters)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/evaluators)

add_library(core_conversion OBJECT
            conversion.cpp
            conversion_ignorelist.cpp)
 target_link_libraries(core_conversion
                       PUBLIC core_conversion_conversionctx
                       PUBLIC core_conversion_converters
                       PUBLIC core_conversion_evaluators
                       PUBLIC core_conversion_var
                       PUBLIC core_conversion_tensorcontainer
                       PUBLIC core_ir
                       PUBLIC core_util
                       PRIVATE core_common)