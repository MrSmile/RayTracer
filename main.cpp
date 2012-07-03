// main.cpp -- entry point
//

#include "cl-helper.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>

#define uint    cl_uint
#define uint2   cl_uint2
#define float   cl_float
#define float3  cl_float3
#define float4  cl_float4
#include "ray-tracer.h"
#undef uint
#undef uint2
#undef float
#undef float3
#undef float4

using namespace std;



int opencl_error(const char *text, cl_int err)
{
    cerr << text << cl_error_string(err) << endl;  return err;
}

int main(int n, const char **arg)
{
    const cl_uint max_platforms = 8;
    cl_platform_id platform[max_platforms];

    cl_uint platform_count;
    cl_int err = clGetPlatformIDs(max_platforms, platform, &platform_count);
    if(err != CL_SUCCESS)return opencl_error("Cannot get platform list: ", err);
    if(platform_count > max_platforms)platform_count = max_platforms;
    if(n <= 1)
    {
        for(cl_uint i = 0; i < platform_count; i++)
        {
            char buf[256];
            err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, sizeof(buf), buf, 0);
            if(err != CL_SUCCESS)return opencl_error("Cannot get platform info: ", err);
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
    if(err != CL_SUCCESS)return opencl_error("Cannot get device: ", err);

    CLContext context = clCreateContext(0, 1, &device, 0, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create context: ", err);

    CLQueue queue = clCreateCommandQueue(context, device, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create command queue: ", err);

    const char *src = "#include \"ray-tracer.cl\"";
    CLProgram program = clCreateProgramWithSource(context, 1, &src, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create program: ", err);

    bool fail = false;  const cl_uint unit_width = 256;
    err = clBuildProgram(program, 1, &device, "-DUNIT_WIDTH=256 -cl-nv-verbose", 0, 0);
    if(err == CL_BUILD_PROGRAM_FAILURE)fail = true;
    else if(err != CL_SUCCESS)return opencl_error("Cannot build program: ", err);

    char buf[65536];
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buf), buf, 0);
    if(err != CL_SUCCESS)return opencl_error("Cannot get build info: ", err);
    // The OpenCL Specification, version 1.1, revision 44 (6/1/11), section 4.1, page 33 footnote:
    // A null terminated string is returned by OpenCL query function calls if the return type of the information being
    // queried is a char[].

    cout << "Build log:\n" << buf << endl;
    if(fail)
    {
        cout << "Compilation failed!" << endl;  return 1;
    }
    cout << "Compilation successfull." << endl;


    GlobalData data;  data.cur_pixel = 0;
    data.group_count = unit_width;  data.ray_count = 8 * unit_width;

    const int width = 512, height = 512;
    data.cam.eye.s[0] = 0;  data.cam.eye.s[1] = -5;  data.cam.eye.s[2] = 0;
    data.cam.top_left.s[0] = -0.5;  data.cam.top_left.s[1] = 1;  data.cam.top_left.s[2] = 0.5;
    data.cam.dx.s[0] = 1.0 / width;  data.cam.dx.s[1] = 0;  data.cam.dx.s[2] = 0;
    data.cam.dy.s[0] = 0;  data.cam.dy.s[1] = 0;  data.cam.dy.s[2] = -1.0 / height;
    data.cam.width = width;  data.cam.height = height;
    data.cam.root_group = 1;  data.cam.root_local = 0;

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

    Group grp[1];
    grp[0].transform_id = tr_none;
    grp[0].shader_id = sh_sky;

    grp[1].transform_id = tr_identity;
    grp[1].shader_id = sh_mesh;
    grp[1].mesh.vtx_offs = 0;
    grp[1].mesh.tri_offs = 0;
    grp[1].mesh.tri_count = 2 * N;
    grp[1].mesh.material_id = 2;

    grp[2].transform_id = tr_none;
    grp[2].shader_id = sh_material;


    CLBuffer global = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeof(data), &data, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create buffer \"global \": ", err);

    CLBuffer area = clCreateBuffer(context, 0, width * height * sizeof(cl_float4), 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create buffer \"area \": ", err);

    CLBuffer grp_data = clCreateBuffer(context, 0, data.group_count * sizeof(GroupData), 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create buffer \"grp_data \": ", err);

    CLBuffer ray_list = clCreateBuffer(context, 0, data.ray_count * sizeof(RayQueue), 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create buffer \"ray_list \": ", err);

    CLBuffer grp_list = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(grp), grp, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create buffer \"grp_list \": ", err);

    CLBuffer mat_list = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create buffer \"mat_list \": ", err);

    CLBuffer aabb_list = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create buffer \"aabb_list \": ", err);

    CLBuffer vtx_list = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(vtx), vtx, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create buffer \"vtx_list \": ", err);

    CLBuffer mesh = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(tri), tri, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create buffer \"mesh \": ", err);

    cl_image_format img_fmt;  img_fmt.image_channel_order = CL_ARGB;  img_fmt.image_channel_data_type = CL_UNORM_INT8;
    CLBuffer image = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &img_fmt, width, height, 0, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create image: ", err);


    return 0;
}
