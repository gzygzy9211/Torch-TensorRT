#include <algorithm>

#include <cuda_runtime.h>
#include "NvInfer.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"
#include "c10/core/DeviceGuard.h"
#include "c10/core/StreamGuard.h"
#include "c10/cuda/CUDAStream.h"

#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

typedef enum { ABI_TARGET_IDX = 0, NAME_IDX, DEVICE_IDX, ENGINE_IDX } SerializedInfoIndex;

std::string slugify(std::string s) {
  std::replace(s.begin(), s.end(), '.', '_');
  return s;
}

TRTEngine::TRTEngine(std::string serialized_engine, CudaDevice cuda_device) {
  std::string _name = "deserialized_trt";
  new (this) TRTEngine(_name, serialized_engine, cuda_device);
}

TRTEngine::TRTEngine(std::vector<std::string> serialized_info) {
  TORCHTRT_CHECK(
      serialized_info.size() == ENGINE_IDX + 1,
      "Program to be deserialized targets an incompatible Torch-TensorRT ABI");
  TORCHTRT_CHECK(
      serialized_info[ABI_TARGET_IDX] == ABI_VERSION,
      "Program to be deserialized targets a different Torch-TensorRT ABI Version ("
          << serialized_info[ABI_TARGET_IDX] << ") than the Torch-TensorRT Runtime ABI Version (" << ABI_VERSION
          << ")");
  std::string _name = serialized_info[NAME_IDX];
  std::string engine_info = serialized_info[ENGINE_IDX];

  CudaDevice cuda_device = deserialize_device(serialized_info[DEVICE_IDX]);
  new (this) TRTEngine(_name, engine_info, cuda_device);
}

TRTEngine::TRTEngine(std::string mod_name, std::string serialized_engine, CudaDevice cuda_device) {
  auto most_compatible_device = get_most_compatible_device(cuda_device);
  TORCHTRT_CHECK(most_compatible_device, "No compatible device was found for instantiating TensorRT engine");
  device_info = most_compatible_device.value();
  set_cuda_device(device_info);

  rt = make_trt(nvinfer1::createInferRuntime(util::logging::get_logger()));

  if (device_info.device_type == nvinfer1::DeviceType::kGPU) {
    allocator = std::make_shared<TorchAllocator>(device_info);
    // if (allocator.get() != nullptr) {
    //   rt->setGpuAllocator(allocator.get());
    // }
  } else {
    allocator.reset((TorchAllocator*)nullptr);
  }
  LOG_DEBUG("Runtime created");

  name = slugify(mod_name);

  cuda_engine = make_trt(rt->deserializeCudaEngine(serialized_engine.c_str(), serialized_engine.size()));
  TORCHTRT_CHECK((cuda_engine.get() != nullptr), "Unable to deserialize the TensorRT engine");
  LOG_DEBUG("Engine loaded");

  exec_ctx = make_trt(allocator.get() == nullptr ?
    cuda_engine->createExecutionContext() :
    cuda_engine->createExecutionContextWithoutDeviceMemory());
  // exec_ctx = make_trt(cuda_engine->createExecutionContext());
  LOG_DEBUG("Context created");

  uint64_t inputs = 0;
  uint64_t outputs = 0;

  for (int64_t x = 0; x < cuda_engine->getNbBindings(); x++) {
    std::string bind_name = cuda_engine->getBindingName(x);
    std::string idx_s = bind_name.substr(bind_name.find("_") + 1);
    uint64_t idx = static_cast<uint64_t>(std::stoi(idx_s));

    if (cuda_engine->bindingIsInput(x)) {
      inputs++;
      in_binding_map[x] = idx;
    } else {
      outputs++;
      out_binding_map[x] = idx;
    }
  }
  num_io = std::make_pair(inputs, outputs);
}

TRTEngine& TRTEngine::operator=(const TRTEngine& other) {
  rt = other.rt;
  cuda_engine = other.cuda_engine;
  device_info = other.device_info;
  exec_ctx = other.exec_ctx;
  num_io = other.num_io;
  return (*this);
}

TRTEngine::~TRTEngine() {
  LOG_DEBUG("[TRTEngine] Deconstructing");
  exec_ctx.reset();
  LOG_DEBUG("[TRTEngine] Context Released");
  cuda_engine.reset();
  LOG_DEBUG("[TRTEngine] Engine Released");
  rt.reset();
  LOG_DEBUG("[TRTEngine] Runtime Released");
  allocator.reset();
  LOG_DEBUG("[TRTEngine] Allocator (if exists) Released");
}

TorchAllocator::TorchAllocator(CudaDevice device) : stream_(c10::cuda::getDefaultCUDAStream()) {
  TORCHTRT_ASSERT(device.device_type != nvinfer1::DeviceType::kDLA, "we do not support DLA device for now");
  c10::DeviceGuard guard(c10::Device(c10::DeviceType::CUDA, device.id));
  auto c10_stream = c10::cuda::getCurrentCUDAStream();
  stream_ = c10_stream;
  device_ = CudaDevice(c10_stream.device().index(), nvinfer1::DeviceType::kGPU);
}

CudaDevice TorchAllocator::get_device_id() const { return device_; }
c10::cuda::CUDAStream& TorchAllocator::get_stream() { return stream_; }
at::cuda::CUDAEvent& TorchAllocator::get_event() { return event_; }
cudaStream_t TorchAllocator::get_cuda_stream() const { return stream_.stream(); }

inline static std::string PTR_STR(void* ptr) { std::ostringstream os; os << ptr; return os.str();}

void* TorchAllocator::allocate(uint64_t const size, uint64_t const alignment, nvinfer1::AllocatorFlags const flags) noexcept {
  c10::StreamGuard guard(stream_);
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    LOG_DEBUG("[TorchAllocator] Allocating bytes: " << size);
    auto blob = at::empty({int64_t(size + alignment)},
                          c10::TensorOptions(c10::kByte).device(stream_.device()));
    auto ptr = blob.data_ptr();
    auto mis_align = (uint64_t)ptr % alignment;
    ptr = (uint8_t*)ptr + alignment - mis_align;
    LOG_DEBUG("[TorchAllocator] Allocated at: " + PTR_STR(ptr));
    blobs_.emplace(ptr, blob);
    LOG_DEBUG("[TorchAllocator] Before return, use count = " << blob.use_count());
    return ptr;
  } catch (...) {
    return nullptr;
  }
}

void TorchAllocator::free(void* const memory) noexcept {
  LOG_DEBUG("[TorchAllocator] Try Freeing bytes at: " + PTR_STR(memory));
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = blobs_.find(memory);
  if (it != blobs_.end()) {
    LOG_DEBUG("[TorchAllocator] Freeing " << it->second.numel() << " bytes (with alignment) at: " <<
      PTR_STR(memory) << ", use count = " << it->second.use_count());
    blobs_.erase(it);
  }
}

// TODO: Implement a call method
// c10::List<at::Tensor> TRTEngine::Run(c10::List<at::Tensor> inputs) {
//     auto input_vec = inputs.vec();
//    auto output_vec = RunCudaEngine(exec_ctx, num_io, input_vec);
//
//     return c10::List<at::Tensor>(output_vec);
// }

namespace {
static auto TORCHTRT_UNUSED TRTEngineTSRegistrtion =
    torch::class_<TRTEngine>("tensorrt", "Engine")
        .def(torch::init<std::vector<std::string>>())
        // TODO: .def("__call__", &TRTEngine::Run)
        // TODO: .def("run", &TRTEngine::Run)
        .def_pickle(
            [](const c10::intrusive_ptr<TRTEngine>& self) -> std::vector<std::string> {
              // Serialize TensorRT engine
              auto serialized_trt_engine = self->cuda_engine->serialize();

              // Adding device info related meta data to the serialized file
              auto trt_engine = std::string((const char*)serialized_trt_engine->data(), serialized_trt_engine->size());

              std::vector<std::string> serialize_info;
              serialize_info.resize(ENGINE_IDX + 1);

              serialize_info[ABI_TARGET_IDX] = ABI_VERSION;
              serialize_info[NAME_IDX] = self->name;
              serialize_info[DEVICE_IDX] = serialize_device(self->device_info);
              serialize_info[ENGINE_IDX] = trt_engine;
              return serialize_info;
            },
            [](std::vector<std::string> seralized_info) -> c10::intrusive_ptr<TRTEngine> {
              return c10::make_intrusive<TRTEngine>(std::move(seralized_info));
            });
} // namespace

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
