// main.cpp -- entry point
//

#include "cl-helper.h"
#include "layout.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>

using namespace std;



int opencl_error(const char *func, cl_int err)
{
    cerr << "OpenCL error in function " << func << ": " << cl_error_string(err) << endl;  return err;
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
    err = clBuildProgram(program, 1, &device, "-cl-nv-verbose", 0, 0);
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


    const float pi = 3.14159265358979323846264338327950288;
    const int N = 16;  Vertex vtx[2 * N];  cl_uint tri[2 * N];
    for(int i = 0; i < N; i++)
    {
        cl_uint dn = 2 * i, up = dn + 1;
        vtx[dn].norm.s[0] = vtx[up].norm.s[0] = vtx[dn].pos.s[0] = vtx[up].pos.s[0] = cos(i * (2 * pi / N));
        vtx[dn].norm.s[1] = vtx[up].norm.s[1] = vtx[dn].pos.s[1] = vtx[up].pos.s[1] = sin(i * (2 * pi / N));
        vtx[dn].norm.s[2] = vtx[up].norm.s[2] = 0;  vtx[dn].pos.s[2] = -0.5f;  vtx[up].pos.s[2] = 0.5f;

        cl_uint dn1 = (i + 1) % N * 2, up1 = dn1 + 1;
        tri[dn] = dn | up << 10 | dn1 << 20;  tri[up] = up1 | dn1 << 10 | up << 20;
    }

    Group grp[1];  memset(grp, 0, sizeof(grp));  grp[0].transform_id = tr_identity;
    grp[0].trans.mat[0].s[0] = grp[0].trans.mat[1].s[1] = grp[0].trans.mat[2].s[2] = 1;

    grp[0].shader_id = sh_tri_list;
    grp[0].shader.tri_list.vtx_offs = 0;
    grp[0].shader.tri_list.tri_offs = 0;
    grp[0].shader.tri_list.tri_count = 2 * N;
    grp[0].shader.tri_list.material_id = 0;

    CLBuffer ray_list = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("clCreateBuffer", err);

    CLBuffer grp_list = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(grp), grp, &err);
    if(err != CL_SUCCESS)return opencl_error("clCreateBuffer", err);

    CLBuffer aabb_list = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("clCreateBuffer", err);

    CLBuffer vtx_list = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(vtx), vtx, &err);
    if(err != CL_SUCCESS)return opencl_error("clCreateBuffer", err);

    CLBuffer tri_list = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(tri), tri, &err);
    if(err != CL_SUCCESS)return opencl_error("clCreateBuffer", err);


    return 0;
}
