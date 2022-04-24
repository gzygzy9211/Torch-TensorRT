add_library(core_ir OBJECT
    ir.cpp
    Input.cpp
    StaticParams.cpp
)
target_link_libraries(core_ir PUBLIC core_util PRIVATE core_common)
