add_library(core_partitioning OBJECT
    SegmentedBlock.cpp
    shape_analysis.cpp
    partitioning.cpp
    PartitionInfo.cpp
)
target_link_libraries(core_partitioning
                      PUBLIC core_ir
                      PUBLIC core_lowering
                      PUBLIC core_conversion
                      PRIVATE core_common)
