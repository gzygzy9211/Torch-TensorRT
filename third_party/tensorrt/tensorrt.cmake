ASSERT_EXISTS(TENSORRT_HOME)

if(CMAKE_HOST_WIN32)
    set(LIB_NVINFER "nvinfer.dll")
    set(LIB_NVINFER_PLUGIN "nvinfer_plugin.dll")
    set(IMP_NVINFER "nvinfer.lib")
    set(IMP_NVINFER_PLUGIN "nvinfer_plugin.lib")
else()
    set(LIB_NVINFER "libnvinfer.so")
    set(LIB_NVINFER_PLUGIN "libnvinfer_plugin.so")
endif()

add_library(nvinfer SHARED IMPORTED)
set_target_properties(nvinfer PROPERTIES
                      IMPORTED_LOCATION ${TENSORRT_HOME}/lib/${LIB_NVINFER}
                      INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_HOME}/include)


add_library(nvinfer_plugin SHARED IMPORTED)
set_target_properties(nvinfer_plugin PROPERTIES
                      IMPORTED_LOCATION ${TENSORRT_HOME}/lib/${LIB_NVINFER_PLUGIN}
                      INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_HOME}/include)

if(CMAKE_HOST_WIN32)
    set_target_properties(nvinfer PROPERTIES
                          IMPORTED_IMPLIB ${TENSORRT_HOME}/lib/${IMP_NVINFER})

    set_target_properties(nvinfer_plugin PROPERTIES
                          IMPORTED_IMPLIB ${TENSORRT_HOME}/lib/${IMP_NVINFER_PLUGIN})
endif()
