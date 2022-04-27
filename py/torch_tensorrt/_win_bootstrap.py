import ctypes
import os.path as osp

import torch


if not __file__.endswith('.py'):
    raise ImportError('Cannot call this module directly')


def bootstrap():
    k32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
    assert hasattr(k32, 'AddDllDirectory'), \
        'Windows minimum requirement is Windows 7 with KB2533623'

    k32.AddDllDirectory.restype = ctypes.c_void_p
    k32.AddDllDirectory.argtypes = [ctypes.c_wchar_p]
    k32.LoadLibraryExW.restype = ctypes.c_void_p
    k32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint]
    k32.SetErrorMode.restype = ctypes.c_uint
    k32.SetErrorMode.argtypes = [ctypes.c_uint]

    # combination of LOAD_LIBRARY_SEARCH_APPLICATION_DIR,
    #     LOAD_LIBRARY_SEARCH_SYSTEM32, and LOAD_LIBRARY_SEARCH_USER_DIRS.
    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
    # temporarily added to the beginning of the list of directories that are
    #     searched for the DLL's dependencies. Directories in the standard
    #     search path are not searched
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100

    prev_error_mode = k32.SetErrorMode(0x0001)
    torchtrt_dll_fullpath = f'{osp.dirname(__file__)}/lib/torchtrt.dll'
    deps_to_load = [
        'nvinfer_builder_resource.dll'
    ]

    try:
        path_handle = k32.AddDllDirectory(osp.dirname(torchtrt_dll_fullpath))
        path2_handle = k32.AddDllDirectory(f'{osp.dirname(torch.__file__)}/lib')
        # print(path_handle, path2_handle)
        dll_handle = k32.LoadLibraryExW(torchtrt_dll_fullpath, None,
                                        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                                        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
        # print(dll_handle)
        load_err = ctypes.get_last_error()
        if load_err != 0:
            raise ImportError(f'fail to load component ({load_err})')
        for dll in (f'{osp.dirname(__file__)}/lib/{dll}' for dll in deps_to_load):
            dll_handle = k32.LoadLibraryExW(dll, None,
                                            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                                            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
            # print(dll_handle)
            load_err = ctypes.get_last_error()
            if load_err != 0:
                raise ImportError(f'fail to load component ({load_err})')

    finally:
        k32.SetErrorMode(prev_error_mode)
