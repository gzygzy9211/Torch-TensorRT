if(PYTHON_EXECUTABLE)
    python_command(HAS_PYTORCH "
import sys
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False
sys.stdout.write(str(has_torch))
    ")
else()
    set(HAS_PYTORCH "False")
endif()

message(STATUS "HAS_PYTORCH = ${HAS_PYTORCH}")
ASSERT_EXISTS(CUDNN_HOME)
set(CUDNN_INCLUDE_PATH ${CUDNN_HOME}/include)
set(CUDNN_LIBRARY_PATH ${CUDNN_HOME}/lib/cudnn.lib)

if(HAS_PYTORCH STREQUAL "True")
    python_command(TORCH_CMAKE_PREFIX "import torch.utils, sys; sys.stdout.write(torch.utils.cmake_prefix_path)")
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
    message(STATUS "using torch python package ${TORCH_CMAKE_PREFIX}")
else()
    ASSERT_EXISTS(LIBTORCH_HOME)
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_HOME}/share/cmake")
    message(STATUS "using libtorch ${LIBTORCH_HOME}/share/cmake")
endif()

find_package(Torch REQUIRED)

unset(CUDNN_INCLUDE_PATH)
unset(CUDNN_LIBRARY_PATH)


# foreach(lib "torch;torch_library;torch_cuda;torch_cuda_cpp;torch_cuda_cu;torch_cpu;")
#     message(STATUS "Setting ${lib} Global")
#     set_target_properties(${lib} PROPERTIES IMPORTED_GLOBAL True)
# endforeach()
