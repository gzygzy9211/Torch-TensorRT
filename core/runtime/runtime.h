#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include "ATen/core/function_schema.h"
#include "NvInfer.h"
#include "core/util/prelude.h"
#include "torch/custom_class.h"
#include "ATen/Tensor.h"
#include "c10/cuda/CUDAStream.h"
#if defined(_WIN32) && defined(C10_BUILD_SHARED_LIBS)
#undef TORCH_CUDA_CPP_API
#define TORCH_CUDA_CPP_API
#define FIX_CUDA_EVENT_INLINE
#endif // defined(_WIN32) && defined(C10_BUILD_SHARED_LIBS)
#include "ATen/cuda/CUDAEvent.h"
#ifdef FIX_CUDA_EVENT_INLINE
#undef TORCH_CUDA_CPP_API
#define TORCH_CUDA_CPP_API C10_IMPORT
#undef FIX_CUDA_EVENT_INLINE
#endif // FIX_CUDA_EVENT_INLINE

namespace torch_tensorrt {
namespace core {
namespace runtime {

using EngineID = int64_t;
const std::string ABI_VERSION = "3";

struct TORCHTRT_CORE_API CudaDevice {
  int64_t id; // CUDA device id
  int64_t major; // CUDA compute major version
  int64_t minor; // CUDA compute minor version
  nvinfer1::DeviceType device_type;
  std::string device_name;

  CudaDevice();
  CudaDevice(int64_t gpu_id, nvinfer1::DeviceType device_type);
  CudaDevice(std::string serialized_device_info);
  CudaDevice& operator=(const CudaDevice& other);
  std::string serialize();
  std::string getSMCapability() const;
  friend std::ostream& operator<<(std::ostream& os, const CudaDevice& device);
};

TORCHTRT_CORE_API void set_cuda_device(CudaDevice& cuda_device);
// Gets the current active GPU (DLA will not show up through this)
TORCHTRT_CORE_API CudaDevice get_current_device();

c10::optional<CudaDevice> get_most_compatible_device(const CudaDevice& target_device);
std::vector<CudaDevice> find_compatible_devices(const CudaDevice& target_device);

std::string serialize_device(CudaDevice& cuda_device);
CudaDevice deserialize_device(std::string device_info);

class TorchAllocator : public nvinfer1::IGpuAllocator {
public:
  virtual void* allocate(uint64_t const size, uint64_t const alignment, nvinfer1::AllocatorFlags const flags) noexcept;
  virtual void free(void* const memory) noexcept;
  c10::cuda::CUDAStream& get_stream();
  at::cuda::CUDAEvent& get_event();
  cudaStream_t get_cuda_stream() const;
  CudaDevice get_device_id() const;
  TorchAllocator(CudaDevice device);
private:
  c10::cuda::CUDAStream stream_;
  CudaDevice device_;
  std::map<void*, at::Tensor> blobs_;
  std::mutex mutex_;
  at::cuda::CUDAEvent event_;
};

struct TORCHTRT_CORE_API TRTEngine : torch::CustomClassHolder {
  // Each engine needs it's own runtime object
  std::shared_ptr<nvinfer1::IRuntime> rt;
  std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine;
  std::shared_ptr<nvinfer1::IExecutionContext> exec_ctx;
  std::pair<uint64_t, uint64_t> num_io;
  std::string name;
  std::mutex mu;
  CudaDevice device_info;
  std::shared_ptr<TorchAllocator> allocator;

  std::unordered_map<uint64_t, uint64_t> in_binding_map;
  std::unordered_map<uint64_t, uint64_t> out_binding_map;

  virtual ~TRTEngine();  // need to make sure allocator is the last member to be destructed
  TRTEngine(std::string serialized_engine, CudaDevice cuda_device);
  TRTEngine(std::vector<std::string> serialized_info);
  TRTEngine(std::string mod_name, std::string serialized_engine, CudaDevice cuda_device);
  TRTEngine& operator=(const TRTEngine& other);
  // TODO: Implement a call method
  // c10::List<at::Tensor> Run(c10::List<at::Tensor> inputs);
};

TORCHTRT_CORE_API std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine);

class DeviceList {
  using DeviceMap = std::unordered_map<int, CudaDevice>;
  DeviceMap device_list;

 public:
  // Scans and updates the list of available CUDA devices
  DeviceList();

 public:
  void insert(int device_id, CudaDevice cuda_device);
  CudaDevice find(int device_id);
  DeviceMap get_devices();
  std::string dump_list();
};

DeviceList get_available_device_list();
const std::unordered_map<std::string, std::string>& get_dla_supported_SMs();

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
