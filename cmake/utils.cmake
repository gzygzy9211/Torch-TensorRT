macro(ASSERT_EXISTS var)
    if(NOT ${var})
        message(FATAL_ERROR "Variable ${var} should be provided by -D${var}=...")
    endif()
endmacro(ASSERT_EXISTS)

macro(python_command output_var cmd)
    ASSERT_EXISTS(PYTHON_EXECUTABLE)
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "${cmd}"
                    RESULT_VARIABLE _retval
                    OUTPUT_VARIABLE ${output_var}
                    ERROR_VARIABLE _error)
    if (NOT _retval EQUAL 0)
        message(FATAL_ERROR "Fail to exec command \"${cmd}\": ${_retval}\n${_error}\n${output_var}")
    else()
        unset(_retval)
        unset(_error)
    endif()
endmacro(python_command)

macro(target_link_libraries_recursive _target)

    set(dependents "${ARGN}")
    # message(STATUS "target = ${_target} deps = ${dependents}")
    foreach(dep ${dependents})
        # skip gen-exp
        if (NOT TARGET ${dep})
            continue()
        endif()
        target_link_libraries(${_target} ${dep})

        # skip non object-library
        get_target_property(deptype ${dep} TYPE)
        if (NOT ${deptype} STREQUAL "OBJECT_LIBRARY")
            continue()
        endif()

        # skip if no further dependency
        get_target_property(depdeps ${dep} INTERFACE_LINK_LIBRARIES)
        if(NOT depdeps)
            continue()
        endif()
        list(PREPEND depdeps ${_target})
        target_link_libraries_recursive(${depdeps})
    endforeach()

endmacro()
