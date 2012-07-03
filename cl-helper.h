// cl-helper.h : OpenCL helper functions
//

#pragma once

#include <CL/opencl.h>
#include <cassert>



template<typename T, cl_int (*release)(T)> class AutoReleaser
{
    T val_;

public:
    AutoReleaser() : val_(0)
    {
    }

    AutoReleaser(T val) : val_(val)
    {
    }

    ~AutoReleaser()
    {
        if(val_)release(val_);
    }

    T attach(T val)
    {
        assert(!val_);  return val_ = val;
    }

    T detach()
    {
        T old = val_;  val_ = 0;  return old;
    }

    T operator = (const T &val)
    {
        return attach(val);
    }

    const T &value() const
    {
        return val_;
    }

    T &value()
    {
        return val_;
    }

    operator T () const
    {
        return val_;
    }
};

typedef AutoReleaser<cl_context, clReleaseContext> CLContext;
typedef AutoReleaser<cl_command_queue, clReleaseCommandQueue> CLQueue;
typedef AutoReleaser<cl_program, clReleaseProgram> CLProgram;
typedef AutoReleaser<cl_mem, clReleaseMemObject> CLBuffer;
typedef AutoReleaser<cl_kernel, clReleaseKernel> CLKernel;


#define CL_ERROR_CASE(err)  case err: return #err;

const char *cl_error_string(cl_int err)
{
    switch(err)
    {
        CL_ERROR_CASE(CL_SUCCESS)
        CL_ERROR_CASE(CL_DEVICE_NOT_FOUND)
        CL_ERROR_CASE(CL_DEVICE_NOT_AVAILABLE)
        CL_ERROR_CASE(CL_COMPILER_NOT_AVAILABLE)
        CL_ERROR_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE)
        CL_ERROR_CASE(CL_OUT_OF_RESOURCES)
        CL_ERROR_CASE(CL_OUT_OF_HOST_MEMORY)
        CL_ERROR_CASE(CL_PROFILING_INFO_NOT_AVAILABLE)
        CL_ERROR_CASE(CL_MEM_COPY_OVERLAP)
        CL_ERROR_CASE(CL_IMAGE_FORMAT_MISMATCH)
        CL_ERROR_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED)
        CL_ERROR_CASE(CL_BUILD_PROGRAM_FAILURE)
        CL_ERROR_CASE(CL_MAP_FAILURE)
        CL_ERROR_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET)
        CL_ERROR_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)

        CL_ERROR_CASE(CL_INVALID_VALUE)
        CL_ERROR_CASE(CL_INVALID_DEVICE_TYPE)
        CL_ERROR_CASE(CL_INVALID_PLATFORM)
        CL_ERROR_CASE(CL_INVALID_DEVICE)
        CL_ERROR_CASE(CL_INVALID_CONTEXT)
        CL_ERROR_CASE(CL_INVALID_QUEUE_PROPERTIES)
        CL_ERROR_CASE(CL_INVALID_COMMAND_QUEUE)
        CL_ERROR_CASE(CL_INVALID_HOST_PTR)
        CL_ERROR_CASE(CL_INVALID_MEM_OBJECT)
        CL_ERROR_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        CL_ERROR_CASE(CL_INVALID_IMAGE_SIZE)
        CL_ERROR_CASE(CL_INVALID_SAMPLER)
        CL_ERROR_CASE(CL_INVALID_BINARY)
        CL_ERROR_CASE(CL_INVALID_BUILD_OPTIONS)
        CL_ERROR_CASE(CL_INVALID_PROGRAM)
        CL_ERROR_CASE(CL_INVALID_PROGRAM_EXECUTABLE)
        CL_ERROR_CASE(CL_INVALID_KERNEL_NAME)
        CL_ERROR_CASE(CL_INVALID_KERNEL_DEFINITION)
        CL_ERROR_CASE(CL_INVALID_KERNEL)
        CL_ERROR_CASE(CL_INVALID_ARG_INDEX)
        CL_ERROR_CASE(CL_INVALID_ARG_VALUE)
        CL_ERROR_CASE(CL_INVALID_ARG_SIZE)
        CL_ERROR_CASE(CL_INVALID_KERNEL_ARGS)
        CL_ERROR_CASE(CL_INVALID_WORK_DIMENSION)
        CL_ERROR_CASE(CL_INVALID_WORK_GROUP_SIZE)
        CL_ERROR_CASE(CL_INVALID_WORK_ITEM_SIZE)
        CL_ERROR_CASE(CL_INVALID_GLOBAL_OFFSET)
        CL_ERROR_CASE(CL_INVALID_EVENT_WAIT_LIST)
        CL_ERROR_CASE(CL_INVALID_EVENT)
        CL_ERROR_CASE(CL_INVALID_OPERATION)
        CL_ERROR_CASE(CL_INVALID_GL_OBJECT)
        CL_ERROR_CASE(CL_INVALID_BUFFER_SIZE)
        CL_ERROR_CASE(CL_INVALID_MIP_LEVEL)
        CL_ERROR_CASE(CL_INVALID_GLOBAL_WORK_SIZE)
        CL_ERROR_CASE(CL_INVALID_PROPERTY)

        CL_ERROR_CASE(CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR)
        CL_ERROR_CASE(CL_PLATFORM_NOT_FOUND_KHR)

        default: return "UNKNOWN";
    }
}

#undef CL_ERROR_CASE
