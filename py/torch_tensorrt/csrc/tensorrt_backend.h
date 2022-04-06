#pragma once
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/backends/backend.h"
#include "torch/version.h"
#if ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR <= 8))
#else
#include "torch/csrc/jit/backends/backend_debug_handler.h"
#include "torch/csrc/jit/backends/backend_preprocess.h"
#endif

namespace torch_tensorrt {
namespace torchscript {
namespace backend {

class TensorRTBackend : public torch::jit::PyTorchBackendInterface {
 public:
  explicit TensorRTBackend() {}
  virtual ~TensorRTBackend() = default;

#if ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR <= 8))
  c10::IValue preprocess(c10::IValue mod, c10::impl::GenericDict method_compile_spec) override;
#else
  bool is_available() override {
    return true;
  }
#endif
  c10::impl::GenericDict compile(c10::IValue processed_mod, c10::impl::GenericDict method_compile_spec) override;
  c10::impl::GenericList execute(c10::IValue handle, c10::impl::GenericList inputs) override;
};

} // namespace backend
} // namespace torchscript
} // namespace torch_tensorrt