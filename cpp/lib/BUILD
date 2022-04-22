package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libtorchtrt_plugins.so",
    srcs = [],
    deps = [
        "//core/plugins:torch_tensorrt_plugins"
    ],
    linkstatic = True,
    linkshared = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
)

cc_binary(
    name = "libtorchtrt_runtime.so",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//core/runtime:runtime",
        "//core/plugins:torch_tensorrt_plugins"
    ],
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
)

cc_binary(
    name = "libtorchtrt.so",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//cpp:torch_tensorrt",
    ],
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
)

cc_binary(
    name = "torchtrt.dll",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//cpp:torch_tensorrt",
    ],
)

filegroup(
    name = "torchtrt_import_name",
    srcs = [":torchtrt.dll"],
    output_group = "interface_library",
    visibility = ["//visibility:private"],
)

genrule(
    name = "torchtrt_win_import",
    srcs = [":torchtrt_import_name"],
    outs = ["torchtrt.lib"],
    cmd = "cp $< $@",
)