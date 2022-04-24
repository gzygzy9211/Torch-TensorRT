import os
import sys

if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is not supported by Torch-TensorRT")

import ctypes
import platform
import torch

_IS_WINDOWS = platform.system() == 'Windows'

if _IS_WINDOWS:
    from ._win_bootstrap import bootstrap
    bootstrap()
    del bootstrap

from torch_tensorrt._version import __version__
from torch_tensorrt._compile import *
from torch_tensorrt._util import *
from torch_tensorrt import ts
from torch_tensorrt import ptq
from torch_tensorrt._enums import *
from torch_tensorrt import logging
from torch_tensorrt._Input import Input
from torch_tensorrt._Device import Device


def _register_with_torch():
    trtorch_dir = os.path.dirname(__file__)
    torchtrt_lib = 'torchtrt.dll' if _IS_WINDOWS else 'libtorchtrt.so'
    torch.ops.load_library(f'{trtorch_dir}/lib/{torchtrt_lib}')


_register_with_torch()
