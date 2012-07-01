
#include <CL/opencl.h>
#include <iostream>
#include <cstdlib>

using namespace std;



template<typename T, cl_int (*release)(T)> class AutoReleaser
{
    T val_;

public:
    AutoReleaser(T val) : val_(val)
    {
    }

    ~AutoReleaser()
    {
        if(val_)release(val_);
    }

    operator T () const
    {
        return val_;
    }
};

typedef AutoReleaser<cl_context, clReleaseContext> CLContext;
typedef AutoReleaser<cl_command_queue, clReleaseCommandQueue> CLQueue;
typedef AutoReleaser<cl_program, clReleaseProgram> CLProgram;



int opencl_error(const char *func, cl_int err)
{
    cerr << "OpenCL error in function " << func << ": " << err << endl;  return err;
}

int main(int n, const char **arg)
{
    const cl_uint max_platforms = 8;
    cl_platform_id platform[max_platforms];

    cl_uint platform_count;
    cl_int err = clGetPlatformIDs(max_platforms, platform, &platform_count);
    if(err != CL_SUCCESS)return opencl_error("clGetPlatformIDs", err);
    if(platform_count > max_platforms)platform_count = max_platforms;
    if(n <= 1)
    {
        for(cl_uint i = 0; i < platform_count; i++)
        {
            char buf[256];
            err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, sizeof(buf), buf, 0);
            if(err != CL_SUCCESS)return opencl_error("clGetPlatformInfo", err);
            cout << "Platform " << i << ": " << buf << endl;
        }
        cout << "Rerun program with platform argument." << endl;  return 0;
    }

    cl_uint index = atoi(arg[1]);
    if(index >= platform_count)
    {
        cout << "Invalid platform index!" << endl;  return 1;
    }

    cl_device_id device;
    err = clGetDeviceIDs(platform[index], CL_DEVICE_TYPE_GPU, 1, &device, 0);
    if(err != CL_SUCCESS)return opencl_error("clGetDeviceIDs", err);

    CLContext context = clCreateContext(0, 1, &device, 0, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("clCreateContext", err);

    CLQueue queue = clCreateCommandQueue(context, device, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("clCreateCommandQueue", err);

    const char *src = "#include \"ray-tracer.cl\"";
    CLProgram program = clCreateProgramWithSource(context, 1, &src, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("clCreateProgramWithSource", err);

    bool fail = false;
    err = clBuildProgram(program, 1, &device, 0, 0, 0);
    if(err == CL_BUILD_PROGRAM_FAILURE)fail = true;
    else if(err != CL_SUCCESS)return opencl_error("clBuildProgram", err);

    char buf[65536];
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buf), buf, 0);
    if(err != CL_SUCCESS)return opencl_error("clGetProgramBuildInfo", err);
    // The OpenCL Specification, version 1.1, revision 44 (6/1/11), section 4.1, page 33 footnote:
    // A null terminated string is returned by OpenCL query function calls if the return type of the information being
    // queried is a char[].

    cout << "Build log:\n" << buf << endl;
    if(fail)
    {
        cout << "Compilation failed!" << endl;  return 1;
    }
    cout << "Compilation successfull." << endl;

    return 0;
}
