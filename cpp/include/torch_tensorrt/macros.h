/*
 * Copyright (c) NVIDIA Corporation.
 * All rights reserved.
 *
 * This library is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#if defined(_MSC_VER)
#if defined(torch_tensorrt_EXPORT)
#define TORCHTRT_API __declspec(dllexport)
#else
#define TORCHTRT_API __declspec(dllimport)
#endif
#define TORCHTRT_HIDDEN
#elif defined(__GNUC__)
#define TORCHTRT_API __attribute__((__visibility__("default")))
#define TORCHTRT_HIDDEN __attribute__((__visibility__("hidden")))
#else
#define TORCHTRT_API
#define TORCHTRT_HIDDEN
#endif // defined(__GNUC__)

// Does this need to be gaurded or something?
#define XSTR(x) #x
#define STR(x) XSTR(x)

#define TORCH_TENSORRT_MAJOR_VERSION 1
#define TORCH_TENSORRT_MINOR_VERSION 0
#define TORCH_TENSORRT_PATCH_VERSION 0
#define TORCH_TENSORRT_VERSION      \
  STR(TORCH_TENSORRT_MAJOR_VERSION) \
  "." STR(TORCH_TENSORRT_MINOR_VERSION) "." STR(TORCH_TENSORRT_PATCH_VERSION)

// Setup namespace aliases for ease of use
namespace torch_tensorrt {
namespace torchscript {}
namespace ts = torchscript;
} // namespace torch_tensorrt
namespace torchtrt = torch_tensorrt;