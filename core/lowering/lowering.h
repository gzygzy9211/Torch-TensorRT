#pragma once
#include <memory>
#include "torch/csrc/jit/ir/ir.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {

struct LowerInfo {
  // Internal flag to ensure torch.jit.Module does not get freezed in lowering.cpp. This is required for QAT models.
  bool unfreeze_module = false;
  // CommonSubexpressionElimination removes duplicate expressions which are used frequently in the graph.
  // for eg:  CSE replaces similar value-d stride nodes of multiple conv layers in a network with a single stride node.
  // In QAT models, if two conv layers are consuming same input, there is a QDQ node for each input of the conv.
  // Since these QDQ nodes will be identical as they share same input, one of them is eliminated due to CSE lowering
  // pass. Disable this in order to not disturb TensorRT's QAT optimizations.
  bool disable_cse = false;
  std::vector<std::string> forced_fallback_modules;
  friend std::ostream& operator<<(std::ostream& os, const LowerInfo& l);
};

struct OutputLayout {
  enum class Type {
    Elem = 0,
    Tuple = 1,
    List = 2,
  };
  Type type;
  torch::jit::Value* self;
  std::vector<OutputLayout> elements;

  std::vector<torch::jit::Value*> outputs_value() const;
};

OutputLayout FlattenOutputs(std::shared_ptr<torch::jit::Graph>& g);

void LowerBlock(torch::jit::Block* b);
void LowerGraph(std::shared_ptr<torch::jit::Graph>& g, LowerInfo lower_info);
torch::jit::Module LowerModule(
    const torch::jit::Module& mod,
    std::string method_name,
    std::unordered_set<std::string> forced_fallback_modules);
std::tuple<std::shared_ptr<torch::jit::Graph>, std::vector<torch::jit::IValue>, OutputLayout> Lower(
    const torch::jit::Module& mod,
    std::string method_name,
    const LowerInfo& lower_info);

} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
