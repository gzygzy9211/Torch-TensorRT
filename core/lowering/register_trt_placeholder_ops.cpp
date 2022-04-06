#include "torch/csrc/jit/runtime/custom_operator.h"
#include <torch/version.h>

namespace torch {
namespace jit {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

RegisterOperators trt_placeholder_ops_reg({
    /// Op marks a Tensor to be conveted from an Torch Tensor
    /// to a TRT constant Tensor
    Operator(
        "trt::const(Tensor val) -> Tensor",
        #if ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR <= 8))
        [](Stack* stack) { /*noop*/ },
        #else
        [](Stack& stack) { /*noop*/ },
        #endif
        aliasAnalysisFromSchema()),
});

} // namespace jit
} // namespace torch
