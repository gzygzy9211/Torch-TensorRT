add_library(core_runtime OBJECT
    CudaDevice.cpp
    DeviceList.cpp
    TRTEngine.cpp
    register_trt_op.cpp
    runtime.cpp
)
target_link_libraries(core_runtime
                      PUBLIC core_util
                      PUBLIC core_plugins
                      PRIVATE core_common)
