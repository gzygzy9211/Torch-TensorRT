add_library(core_conversion_tensorcontainer OBJECT
            TensorContainer.cpp)
target_link_libraries(core_conversion_tensorcontainer
                      PUBLIC core_util
                      PRIVATE core_common)
