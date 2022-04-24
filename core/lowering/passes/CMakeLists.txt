add_library(core_lowering_passes OBJECT
    convNd_to_convolution.cpp
    exception_elimination.cpp
    fuse_addmm_branches.cpp
    linear_to_addmm.cpp
    module_fallback.cpp
    op_aliasing.cpp
    reduce_to.cpp
    remove_bn_dim_check.cpp
    remove_contiguous.cpp
    remove_dropout.cpp
    remove_nops.cpp
    silu_to_sigmoid_multiplication.cpp
    unpack_addmm.cpp
    unpack_batch_norm.cpp
    unpack_log_softmax.cpp
    unpack_hardswish.cpp
    unpack_std.cpp
    unpack_var.cpp
)
target_link_libraries(core_lowering_passes PUBLIC core_util PRIVATE core_common)
