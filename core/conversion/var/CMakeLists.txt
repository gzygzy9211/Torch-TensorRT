add_library(core_conversion_var OBJECT Var.cpp)
target_link_libraries(core_conversion_var
                      PUBLIC core_util
                      PRIVATE core_common)
