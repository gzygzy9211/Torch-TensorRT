add_library(core_conversion_evaluators OBJECT
    NodeEvaluatorRegistry.cpp
    aten.cpp
    eval_util.cpp
    prim.cpp
)
target_link_libraries(core_conversion_evaluators
                      PUBLIC core_util
                      PUBLIC core_conversion_var
                      PUBLIC core_conversion_tensorcontainer
                      PRIVATE core_common)
