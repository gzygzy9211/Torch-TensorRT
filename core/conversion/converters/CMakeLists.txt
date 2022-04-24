add_library(core_conversion_converters OBJECT
    Weights.cpp
    converter_util.cpp
    NodeConverterRegistry.cpp
    impl/activation.cpp
    impl/batch_norm.cpp
    impl/cast.cpp
    impl/concat.cpp
    impl/constant.cpp
    impl/constant_pad.cpp
    impl/conv_deconv.cpp
    impl/cumsum.cpp
    impl/element_wise.cpp
    impl/expand.cpp
    impl/interpolate.cpp
    impl/layer_norm.cpp
    impl/linear.cpp
    impl/lstm_cell.cpp
    impl/matrix_multiply.cpp
    impl/normalize.cpp
    impl/pooling.cpp
    impl/quantization.cpp
    impl/reduce.cpp
    impl/replication_pad.cpp
    impl/select.cpp
    impl/shuffle.cpp
    impl/softmax.cpp
    impl/squeeze.cpp
    impl/stack.cpp
    impl/topk.cpp
    impl/unary.cpp
    impl/unsqueeze.cpp
)
target_link_libraries(core_conversion_converters
                      PUBLIC core_util
                      PUBLIC core_conversion_var
                      PUBLIC core_conversion_tensorcontainer
                      PUBLIC core_conversion_conversionctx
                      PUBLIC core_plugins
                      PRIVATE core_common)
