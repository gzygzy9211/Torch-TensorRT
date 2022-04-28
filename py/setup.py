import os
import sys
import glob
import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install
from distutils.cmd import Command
from wheel.bdist_wheel import bdist_wheel

from torch.utils import cpp_extension
import torch
from shutil import copyfile, rmtree

import subprocess
import platform
import warnings

dir_path = os.path.dirname(os.path.realpath(__file__))

CXX11_ABI = False

JETPACK_VERSION = None

__version__ = '1.0.0'

IS_WINDOWS = platform.system() == 'Windows'


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


if "--release" not in sys.argv:
    __version__ = __version__ + "+" + get_git_revision_short_hash()
else:
    sys.argv.remove("--release")

if "--use-cxx11-abi" in sys.argv:
    sys.argv.remove("--use-cxx11-abi")
    CXX11_ABI = True

if platform.uname().processor == "aarch64":
    if "--jetpack-version" in sys.argv:
        version_idx = sys.argv.index("--jetpack-version") + 1
        version = sys.argv[version_idx]
        sys.argv.remove(version)
        sys.argv.remove("--jetpack-version")
        if version == "4.5":
            JETPACK_VERSION = "4.5"
        elif version == "4.6":
            JETPACK_VERSION = "4.6"
    if not JETPACK_VERSION:
        warnings.warn("Assuming jetpack version to be 4.6, if not use the --jetpack-version option")
        JETPACK_VERSION = "4.6"


TENSORRT_HOME = os.environ.get('TENSORRT_HOME', None)
assert TENSORRT_HOME is not None, 'specify tensorrt path via environment variable TENSORRT_HOME'


CUDNN_HOME = os.environ.get('CUDNN_HOME', None)
assert CUDNN_HOME is not None, 'specify cudnn path via environment variable CUDNN_HOME'


def which(program):
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


BAZEL_EXE = which("bazelisk")

if BAZEL_EXE is None:
    BAZEL_EXE = which("bazel" if not IS_WINDOWS else "bazel.exe")
    if BAZEL_EXE is None and JETPACK_VERSION is not None:
        sys.exit("Could not find bazel in PATH")


CMAKE_EXE = which("cmake" if not IS_WINDOWS else 'cmake.exe')
if CMAKE_EXE is None and JETPACK_VERSION is None:
    sys.exc_info("Could not find cmake in PATH")


PATCH_ELF_EXE = dir_path + '/patchelf'


def cuda_major() -> str:
    from torch.version import cuda
    return cuda.split('.')[0]


def tensorrt_major() -> str:
    return tensorrt_version().split('.')[0]


def _grep_version_integer(path: str, macro: str) -> int:
    import re
    with open(path, 'r') as f:
        ver_line = [line.strip() for line in f
                    if line.startswith(f'#define {macro} ')][0]
        ver_number = int(re.match(f'#define {macro} ([0-9]+)',
                                  ver_line).groups()[0])
        return ver_number


def cudnn_version() -> str:
    header_path = f'{CUDNN_HOME}/include/cudnn_version.h'
    major = _grep_version_integer(header_path, 'CUDNN_MAJOR')
    minor = _grep_version_integer(header_path, 'CUDNN_MINOR')
    patch = _grep_version_integer(header_path, 'CUDNN_PATCHLEVEL')
    return f'{major}.{minor}.{patch}'


def tensorrt_version() -> str:
    header_path = f'{TENSORRT_HOME}/include/NvInferVersion.h'
    major = _grep_version_integer(header_path, 'NV_TENSORRT_MAJOR')
    minor = _grep_version_integer(header_path, 'NV_TENSORRT_MINOR')
    patch = _grep_version_integer(header_path, 'NV_TENSORRT_PATCH')
    return f'{major}.{minor}.{patch}'


def check_cuda_version():
    from torch.version import cuda as torch_cuda_ver

    _ = cpp_extension.library_paths(cuda=True)
    cuda_ver_number = _grep_version_integer(
        cpp_extension.CUDA_HOME + '/include/cuda.h', 'CUDA_VERSION'
    )
    cuda_ver = f'{cuda_ver_number // 1000}.{cuda_ver_number % 100 // 10}'

    assert cuda_ver == torch_cuda_ver, \
        f'cuda version not match: torch {torch_cuda_ver} vs toolkit {cuda_ver}'


def check_tensorrt_version():
    TRT_2_CUDNN = {
        '8.4.0': '8.3.2',
        '8.2.4': '8.2.1',
        '8.2.3': '8.2.1',
        '8.2.2': '8.2.1',
        '8.2.1': '8.2.1',
        '8.2.0': '8.2.1',
        '8.0.3': '8.2.0',
        '8.0.2': '8.2.0',
        '8.0.1': '8.2.0',
        '8.0.0': '8.2.0',
        '7.2.3': '8.1.1',
        '7.2.2': '8.0.5',
        '7.2.1': '8.0.4',
        '7.2.0': '8.0.2',
        '7.1.3': '8.0.0',
        '7.1.2': '8.0.0',
        '7.1.0': '7.6.5',
        '7.0.0': '7.6.5',
    }
    trt = tensorrt_version()
    cudnn = cudnn_version()
    if trt in TRT_2_CUDNN:
        if TRT_2_CUDNN[trt] == cudnn:
            return
        elif tuple(TRT_2_CUDNN[trt].split('.')[0:2]) == tuple(cudnn.split('.')[0:2]):
            print('=============== !! WARNING !! ================')
            print(f'tensorrt {trt} expect cudnn {TRT_2_CUDNN[trt]}, but {cudnn} provided')
            print('dismatch at patch-level')
            print('=============== !! WARNING !! ================')
            return
        else:
            assert TRT_2_CUDNN[trt] == cudnn, \
                f'tensorrt {trt} expect cudnn {TRT_2_CUDNN[trt]}, but {cudnn} provided'
    else:
        print('=============== !! WARNING !! ================')
        print(f'tensorrt {trt} is not recorded, cannot determine its compatibility')
        print(f'with cudnn {cudnn}, build anyway')
        print('=============== !! WARNING !! ================')       


def build_libtorchtrt_pre_cxx11_abi_cmake(develop=True, cxx11_abi=False):
    if JETPACK_VERSION is not None:
        return 
    cmd = [CMAKE_EXE, '-S', f'{dir_path}/..', '-B', f'{dir_path}/core_build/']
    cmd.append(f'-DTENSORRT_HOME={TENSORRT_HOME}')
    cmd.append(f'-DCUDNN_HOME={CUDNN_HOME}')
    cmd.append(f'-DPYTHON_EXECUTABLE={sys.executable}')
    cmd.append(f'-DCMAKE_CUDA_COMPILER={cpp_extension.CUDA_HOME}/bin/nvcc')

    if not IS_WINDOWS:
        if cxx11_abi:
            cmd.append('-DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"')
            cmd.append('-DCMAKE_CUDA_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"')
        else:
            cmd.append('-DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"')
            cmd.append('-DCMAKE_CUDA_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"')

    if develop:
        cmd.append('-DTORCHTRT_DEVELOP=1')
        cfg = 'RelWithDebInfo'
    else:
        cfg = 'Release'

    print('configuring libtorchtrt')
    subprocess.check_call(cmd)

    cmd = [CMAKE_EXE, '--build', f'{dir_path}/core_build', '-j', str(os.cpu_count())]
    if IS_WINDOWS:
        cmd.append('--config')
        cmd.append(cfg)

    print('building libtorchtrt')
    subprocess.check_call(cmd)

    cmd = [CMAKE_EXE, '--install', f'{dir_path}/core_build']
    subprocess.check_call(cmd)


def build_libtorchtrt_pre_cxx11_abi(develop=True, use_dist_dir=True, cxx11_abi=False):
    if JETPACK_VERSION is None:
        build_libtorchtrt_pre_cxx11_abi_cmake(develop, cxx11_abi)
        return

    with open(f'{dir_path}/../WORKSPACE', 'r') as tpl, open(f'{dir_path}/../WORKSPACE.bazel', 'w') as wksp:
        replacement = {
            '/path/to/cuda': cpp_extension.CUDA_HOME.replace('\\', '/'),
            '/path/to/tensorrt': TENSORRT_HOME.replace('\\', '/'),
            '/path/to/torch': os.path.dirname(torch.__file__).replace('\\', '/')
        }
        content = tpl.read()
        for key, value in replacement.items():
            content = content.replace(key, value)
        wksp.write(content)

    cmd = [BAZEL_EXE, "build"]
    cmd.append("//:libtorchtrt")
    if develop:
        cmd.append("--compilation_mode=dbg")
    else:
        cmd.append("--compilation_mode=opt")
    if use_dist_dir:
        cmd.append("--distdir=third_party/dist_dir/x86_64-linux-gnu")
    if not cxx11_abi:
        cmd.append("--config=python")
    else:
        print("using CXX11 ABI build")

    if JETPACK_VERSION == "4.5":
        cmd.append("--platforms=//toolchains:jetpack_4.5")
        print("Jetpack version: 4.5")
    elif JETPACK_VERSION == "4.6":
        cmd.append("--platforms=//toolchains:jetpack_4.6")
        print("Jetpack version: 4.6")

    print("building libtorchtrt")
    status_code = subprocess.run(cmd).returncode

    if status_code != 0:
        sys.exit(status_code)


def gen_version_file():
    if not IS_WINDOWS and not os.path.exists(dir_path + '/torch_tensorrt/_version.py'):
        os.mknod(dir_path + '/torch_tensorrt/_version.py')

    with open(dir_path + '/torch_tensorrt/_version.py', 'w') as f:
        print("creating version file")
        f.write("__version__ = \"" + __version__ + '\"')


def copy_libtorchtrt(multilinux=False):
    if not os.path.exists(dir_path + '/torch_tensorrt/lib'):
        os.makedirs(dir_path + '/torch_tensorrt/lib')

    print("copying library into module")
    if multilinux:
        copyfile(dir_path + "/build/libtrtorch_build/libtrtorch.so", dir_path + '/trtorch/lib/libtrtorch.so')
    elif JETPACK_VERSION is not None:
        if os.path.exists(f'{dir_path}/torch_tensorrt/lib'):
            rmtree(f'{dir_path}/torch_tensorrt/lib') \
                if os.path.isdir(f'{dir_path}/torch_tensorrt/lib') \
                else os.remove(f'{dir_path}/torch_tensorrt/lib')
        os.system("tar -xzf ../bazel-bin/libtorchtrt.tar.gz --strip-components=2 -C " + dir_path + "/torch_tensorrt")
    else:
        pass  # cmake has already installed torchtrt to the right place

    if IS_WINDOWS:
        for dll in ['nvinfer.dll', 'nvinfer_plugin.dll', 'nvinfer_builder_resource.dll']:
            copyfile(f'{TENSORRT_HOME}/lib/{dll}', f'{dir_path}/torch_tensorrt/lib/{dll}')
        return

    # TODO: complete windows building
    for lib in os.listdir(f'{dir_path}/torch_tensorrt/lib'):
        subprocess.check_call(['strip', f'{dir_path}/torch_tensorrt/lib/{lib}'])

    print("copying dependent library")
    _ = cpp_extension.library_paths(cuda=True)
    suffix = '.so'
    # topological sort by dependency (ignore thoese provided by torch or by system):
    #   libcublasLt.so -> {}
    #   libcublas.so -> static { libcublasLt.so }
    #   libcudnn_ops_infer.so -> static { libcublas.so, libcublasLt.so }
    #   libcudnn_ops_train.so -> static { libcudnn_ops_infer.so }
    #   libcudnn.so -> dynamic { libcudnn_ops_infer.so, libcudnn_ops_train.so } (only satisfy libnvinfer_plugin)
    #   libnvinfer.so -> dynamic { libcublas.so, libcublasLt.so, libnvinfer_builder_resource.so }
    #                    (if trt == 7.x static { libcudnn.so, libmyelin.so, libcublas.so, libcublasLt.so })
    #   libnvinfer_plugin.so -> static { libnvinfer.so, libcudnn.so, libcublas.so, libcublasLt.so }
    #   libtorchtrt*.so -> static { libnvinfer.so, libnvinfer_plugin.so }
    #
    #   NOTE: for trt 7.x, libnvinfer will be dependent on the whole cudnn library   
    # skip cudart because we always improt torch first and torch can provides cudart

    # cublas & cublasLt
    cublas_libs = [f'{cpp_extension.CUDA_HOME}/lib64/{file}'
                   for file in os.listdir(cpp_extension.CUDA_HOME + '/lib64')
                   if file.endswith(suffix) and 'cublas' in file]
    assert (len(cublas_libs) == 2 and
            any((f'cublas{suffix}' in lib for lib in cublas_libs)) and
            any((f'cublasLt{suffix}' in lib for lib in cublas_libs))), str(cublas_libs)
    for lib in cublas_libs:
        # dst_link = f'{dir_path}/torch_tensorrt/lib/{os.path.basename(lib)}'
        dst_entity = f'{dir_path}/torch_tensorrt/lib/{os.path.basename(lib)}.{cuda_major()}'
        copyfile(lib, dst_entity)

    # cudnn
    cudnn_lib = f'{CUDNN_HOME}/lib/libcudnn{suffix}'
    cudnn_components = [f'{CUDNN_HOME}/lib/{file}' for file in os.listdir(f'{CUDNN_HOME}/lib')
                        if file.endswith(suffix) and
                        ('_ops_' in file or int(tensorrt_major()) <= 7)]
    assert os.path.exists(cudnn_lib)
    assert (len(cudnn_components) >= 2 and
            any((f'cudnn_ops_infer{suffix}' in lib for lib in cudnn_components)) and
            any((f'cudnn_ops_train{suffix}' in lib for lib in cudnn_components)))
    cudnn_libs_mapping = {}
    for lib in cudnn_components:
        ver_major = subprocess.check_output(['realpath', lib]).decode('utf8').split('.')[-3]
        dst_entity = f'{dir_path}/torch_tensorrt/lib/{os.path.basename(lib)}.{ver_major}'
        subprocess.check_call([PATCH_ELF_EXE, '--output', dst_entity, '--set-rpath', '$ORIGIN', lib])
    for lib in (cudnn_lib,):
        ver_major = subprocess.check_output(['realpath', lib]).decode('utf8').split('.')[-3]
        sha256 = subprocess.check_output(['sha256sum', lib]).decode('utf8')[0:8]
        libname = os.path.splitext(os.path.basename(lib))[0]
        dst_entity = f'{dir_path}/torch_tensorrt/lib/{libname}-{sha256}{suffix}.{ver_major}'
        subprocess.check_call([PATCH_ELF_EXE, '--output', dst_entity, '--set-rpath', '$ORIGIN', lib])
        cudnn_libs_mapping[os.path.basename(f'{lib}.{ver_major}')] = os.path.basename(dst_entity)

    replace_needed_cudnn = sum((['--replace-needed', key, value]
                                for key, value in cudnn_libs_mapping.items()), [])

    # nvinfer
    nvinfer_libs = [f'{TENSORRT_HOME}/lib/{line}' for line in os.listdir(f'{TENSORRT_HOME}/lib/')
                    if suffix in line and 'parser' not in line and
                    not os.path.islink(f'{TENSORRT_HOME}/lib/{line}')]
    nvinfer_libs_mapping = {}
    assert (len(nvinfer_libs) >= 2 and
            any((f'nvinfer{suffix}' in lib for lib in nvinfer_libs)) and
            any((f'nvinfer_plugin{suffix}' in lib for lib in nvinfer_libs)) and 
            all((lib.rsplit('.', 3)[0].endswith(suffix)
                 for lib in nvinfer_libs))), str(nvinfer_libs)
    for lib in nvinfer_libs:
        # for lib as a static dependency, empirically it has a symlink without version suffix
        if not os.path.exists(lib.rsplit('.', 3)[0]):
            copyfile(lib, f'{dir_path}/torch_tensorrt/lib/{os.path.basename(lib)}')
            continue
        ver_major = subprocess.check_output(['realpath', lib]).decode('utf8').split('.')[-3]
        sha256 = subprocess.check_output(['sha256sum', lib]).decode('utf8')[0:8]
        libname = os.path.basename(lib).rsplit('.', 4)[0]
        dst_entity = f'{dir_path}/torch_tensorrt/lib/{libname}-{sha256}{suffix}.{ver_major}'
        cmd = [PATCH_ELF_EXE, '--output', dst_entity, '--set-rpath', '$ORIGIN', lib]
        cmd += replace_needed_cudnn
        subprocess.check_call(cmd)
        nvinfer_libs_mapping[f'{libname}{suffix}.{ver_major}'] = os.path.basename(dst_entity)
    print(nvinfer_libs_mapping)

    replace_needed_nvinfer = sum((['--replace-needed', key, value]
                                  for key, value in nvinfer_libs_mapping.items()), [])
    for lib in nvinfer_libs_mapping.values():
        cmd = [PATCH_ELF_EXE, f'{dir_path}/torch_tensorrt/lib/{lib}'] + replace_needed_nvinfer
        subprocess.check_call(cmd)

    # patch libtorchtrt
    for lib in ('libtorchtrt.so', 'libtorchtrt_runtime.so', 'libtorchtrt_plugins.so'):
        lib_fullpath = f'{dir_path}/torch_tensorrt/lib/{lib}'
        assert os.path.exists(lib_fullpath)
        subprocess.check_call([f'{dir_path}/patchelf', '--output', lib_fullpath,
                               '--set-rpath', '$ORIGIN'] + replace_needed_nvinfer + [
                               lib_fullpath])


class DevelopCommand(develop):
    description = "Builds the package and symlinks it into the PYTHONPATH"

    def initialize_options(self):
        develop.initialize_options(self)

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        global CXX11_ABI
        check_cuda_version()
        check_tensorrt_version()
        build_libtorchtrt_pre_cxx11_abi(develop=True, cxx11_abi=CXX11_ABI)
        gen_version_file()
        copy_libtorchtrt()
        develop.run(self)


class InstallCommand(install):
    description = "Builds the package"

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        global CXX11_ABI
        check_cuda_version()
        check_tensorrt_version()
        build_libtorchtrt_pre_cxx11_abi(develop=False, cxx11_abi=CXX11_ABI)
        gen_version_file()
        copy_libtorchtrt()
        install.run(self)


class BdistCommand(bdist_wheel):
    description = "Builds the package"

    def initialize_options(self):
        bdist_wheel.initialize_options(self)

    def finalize_options(self):
        bdist_wheel.finalize_options(self)

    def run(self):
        global CXX11_ABI
        check_cuda_version()
        check_tensorrt_version()
        build_libtorchtrt_pre_cxx11_abi(develop=False, cxx11_abi=CXX11_ABI)
        gen_version_file()
        copy_libtorchtrt()
        bdist_wheel.run(self)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    PY_CLEAN_DIRS = [
        './build',
        './dist',
        './torch_tensorrt/__pycache__',
        './torch_tensorrt/lib',
        './torch_tensorrt/include',
        './torch_tensorrt/bin',
        './*.pyc',
        './*.tgz',
        './*.egg-info',
    ]
    PY_CLEAN_FILES = [
        './torch_tensorrt/*.so', './torch_tensorrt/_version.py', './torch_tensorrt/BUILD', './torch_tensorrt/WORKSPACE',
        './torch_tensorrt/LICENSE'
    ]
    description = "Command to tidy up the project root"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path_spec in self.PY_CLEAN_DIRS:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(dir_path):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, dir_path))
                print('Removing %s' % os.path.relpath(path))
                rmtree(path)

        for path_spec in self.PY_CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(dir_path):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, dir_path))
                print('Removing %s' % os.path.relpath(path))
                os.remove(path)


ext_modules = [
    cpp_extension.CUDAExtension(
        'torch_tensorrt._C', [
            'torch_tensorrt/csrc/torch_tensorrt_py.cpp',
            'torch_tensorrt/csrc/tensorrt_backend.cpp',
            'torch_tensorrt/csrc/tensorrt_classes.cpp',
            'torch_tensorrt/csrc/register_tensorrt_classes.cpp',
        ],
        library_dirs=[(dir_path + '/torch_tensorrt/lib/'), "/opt/conda/lib/python3.6/config-3.6m-x86_64-linux-gnu"],
        libraries=["torchtrt"],
        include_dirs=[
            dir_path + "/torch_tensorrt/csrc", dir_path + "/torch_tensorrt/include",
            dir_path + "/torch_tensorrt/include/torch_tensorrt",
            TENSORRT_HOME + "/include",
        ],
        extra_compile_args=[
            "-Wno-deprecated",
            "-Wno-deprecated-declarations",
        ] + (["-D_GLIBCXX_USE_CXX11_ABI=1"] if CXX11_ABI else ["-D_GLIBCXX_USE_CXX11_ABI=0"]),
        extra_link_args=[
            "-Wno-deprecated", "-Wno-deprecated-declarations", "-Wl,--no-as-needed", "-ltorchtrt",
            "-Wl,-rpath,$ORIGIN/lib", "-lpthread", "-ldl", "-lutil", "-lrt", "-lm", "-Xlinker", "-export-dynamic"
        ] + (["-D_GLIBCXX_USE_CXX11_ABI=1"] if CXX11_ABI else ["-D_GLIBCXX_USE_CXX11_ABI=0"]),
        undef_macros=["NDEBUG"])
]

if IS_WINDOWS:
    ext_modules[0].extra_link_args.clear()
    ext_modules[0].extra_compile_args.clear()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='torch_tensorrt',
    version=__version__,
    author='NVIDIA',
    author_email='narens@nvidia.com',
    url='https://nvidia.github.io/torch-tensorrt',
    description=
    'Torch-TensorRT is a package which allows users to automatically compile PyTorch and TorchScript modules to TensorRT while remaining in PyTorch',
    long_description_content_type='text/markdown',
    long_description=long_description,
    ext_modules=ext_modules,
    install_requires=[
        'torch>=1.8.0+cu111<1.9.0',
    ],
    setup_requires=[],
    cmdclass={
        'install': InstallCommand,
        'clean': CleanCommand,
        'develop': DevelopCommand,
        'build_ext': cpp_extension.BuildExtension,
        'bdist_wheel': BdistCommand,
    },
    zip_safe=False,
    license="BSD",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Stable", "Environment :: GPU :: NVIDIA CUDA",
        "License :: OSI Approved :: BSD License", "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", "Operating System :: POSIX :: Linux", "Programming Language :: C++",
        "Programming Language :: Python", "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering", "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development", "Topic :: Software Development :: Libraries"
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={
        'torch_tensorrt': [
            'lib/*', 'include/torch_tensorrt/*.h', 'include/torch_tensorrt/core/*.h',
            'include/torch_tensorrt/core/conversion/*.h', 'include/torch_tensorrt/core/conversion/conversionctx/*.h',
            'include/torch_tensorrt/core/conversion/converters/*.h',
            'include/torch_tensorrt/core/conversion/evaluators/*.h',
            'include/torch_tensorrt/core/conversion/tensorcontainer/*.h',
            'include/torch_tensorrt/core/conversion/var/*.h', 'include/torch_tensorrt/core/ir/*.h',
            'include/torch_tensorrt/core/lowering/*.h', 'include/torch_tensorrt/core/lowering/passes/*.h',
            'include/torch_tensorrt/core/partitioning/*.h', 'include/torch_tensorrt/core/plugins/*.h',
            'include/torch_tensorrt/core/plugins/impl/*.h', 'include/torch_tensorrt/core/runtime/*.h',
            'include/torch_tensorrt/core/util/*.h', 'include/torch_tensorrt/core/util/logging/*.h', 'bin/*', 'BUILD',
            'WORKSPACE'
        ],
    },
    exclude_package_data={
        '': ['*.cpp'],
        'torch_tensorrt': ['csrc/*.cpp'],
    })
