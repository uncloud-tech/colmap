// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#if CUDART_VERSION < 11030 && defined(__linux__)
#include <dlfcn.h>
#endif

namespace colmap::mvs::owned {
// Inventory-only lazy Driver API lookup: never introduce a libcuda DT_NEEDED
// dependency into COLMAP. Construction belongs inside the bounded query, not
// static initialization. The Runtime entry-point query does not create a CUDA
// context; older Linux SDKs use the driver's stable exported symbols instead.
class InventoryCudaDriver {
 public:
  InventoryCudaDriver() = default;
  InventoryCudaDriver(const InventoryCudaDriver&) = delete;
  InventoryCudaDriver& operator=(const InventoryCudaDriver&) = delete;
  ~InventoryCudaDriver() {
#if CUDART_VERSION < 11030 && defined(__linux__)
    if (library_) dlclose(library_);
#endif
  }

  template <typename Function>
  Function Lookup(const char* name, const char* legacy_name = nullptr) {
    void* address = nullptr;
#if CUDART_VERSION >= 11030
#if CUDART_VERSION >= 12050
    const auto status = cudaGetDriverEntryPointByVersion(name, &address, CUDA_VERSION, 0);
#else
    const auto status = cudaGetDriverEntryPoint(name, &address, 0);
#endif
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("CUDA inventory driver lookup failed: ") +
                               name + ": " + cudaGetErrorString(status));
#elif defined(__linux__)
    if (!library_) {
      library_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
      if (!library_)
        throw std::runtime_error("CUDA inventory requires an available NVIDIA driver (libcuda.so.1)");
    }
    address = dlsym(library_, legacy_name ? legacy_name : name);
#else
    throw std::runtime_error("CUDA inventory lazy driver lookup requires CUDA 11.3+ on this platform");
#endif
    if (!address)
      throw std::runtime_error(std::string("CUDA inventory driver entry point unavailable: ") + name);
    return reinterpret_cast<Function>(address);
  }

 private:
#if CUDART_VERSION < 11030 && defined(__linux__)
  void* library_ = nullptr;
#endif
};
}  // namespace colmap::mvs::owned
