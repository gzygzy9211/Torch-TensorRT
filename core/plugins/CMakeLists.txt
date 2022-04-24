add_library(core_plugins OBJECT
    impl/interpolate_plugin.cpp
    impl/normalize_plugin.cpp
    register_plugins.cpp
)
target_link_libraries(core_plugins PUBLIC core_util
                                   PRIVATE core_common
                                   PRIVATE nvinfer_plugin)
