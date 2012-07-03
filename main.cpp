// main.cpp -- entry point
//

#include "cl-helper.h"
#include <SDL/SDL.h>
#include <SDL/SDL_opengl.h>
#include <GL/glx.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>


inline cl_int delete_texture(GLuint tex)
{
    glDeleteTextures(1, &tex);  return 0;
}

//typedef AutoReleaser<SDL_Window *, SDL_DestroyWindow> SDLWindow;
//typedef AutoReleaser<SDL_GL_Context, SDL_DeleteContext> GLContext;
typedef AutoReleaser<GLuint, delete_texture> GLTexture;

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



int sdl_error(const char *text)
{
    cerr << text << SDL_GetError() << endl;  SDL_ClearError();  return 1;
}

int opencl_error(const char *text, cl_int err)
{
    cerr << text << cl_error_string(err) << endl;  return err;
}

int ray_tracer(cl_platform_id platform, cl_device_id device)
{
    /*if(SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3) ||
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2))
            return sdl_error("Failed to set OpenGL version: ");*/

    if(SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 0))return sdl_error("Failed to disable double-buffering: ");
    if(SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0))return sdl_error("Failed to disable depth buffer: ");

    /*SDLWindow window = SDL_CreateWindow("RayTracer 1.0",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 600, 600, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if(!window)return sdl_error("Cannot create window: ");

    GLContext context = SDL_GL_CreateContext(window);
    if(*SDL_GetError())return sdl_error("Cannot create OpenGL context: ");*/

    const int width = 512, height = 512;
    SDL_Surface *surface = SDL_SetVideoMode(width, height, 0, SDL_HWSURFACE | SDL_OPENGL);
    if(!surface)return sdl_error("Cannot create OpenGL context: ");
    SDL_WM_SetCaption("RayTracer 1.0", 0);

    GLTexture texture;  glGenTextures(1, &texture.value());  glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glEnable(GL_TEXTURE_2D);  glColor3f(1, 1, 1);  glViewport(0, 0, 512, 512);


    // OpenCL init

    cl_context_properties prop[] =
    {
        CL_GL_CONTEXT_KHR, cl_context_properties(glXGetCurrentContext()),
        CL_GLX_DISPLAY_KHR, cl_context_properties(glXGetCurrentDisplay()),
        CL_CONTEXT_PLATFORM, cl_context_properties(platform), 0
    };

    cl_int err;
    CLContext context = clCreateContext(prop, 1, &device, 0, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create context: ", err);

    CLQueue queue = clCreateCommandQueue(context, device, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create command queue: ", err);

    const char *src = "#include \"ray-tracer.cl\"";
    CLProgram program = clCreateProgramWithSource(context, 1, &src, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create program: ", err);

    bool fail = false;  const size_t unit_width = 256;
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

    Group grp[3];
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

    const size_t area_size = width * height;
    CLBuffer area = clCreateBuffer(context, 0, area_size * sizeof(cl_float4), 0, &err);
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

    CLBuffer tri_list = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(tri), tri, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create buffer \"tri_list \": ", err);

    CLBuffer image = clCreateFromGLTexture2D(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texture, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create image: ", err);


    CLKernel init_groups = clCreateKernel(program, "init_groups", &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create kernel \"init_groups\": ", err);
    err = clSetKernelArg(init_groups, 0, sizeof(cl_mem), &grp_data.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 0 for kernel \"init_groups\": ", err);

    CLKernel init_rays = clCreateKernel(program, "init_rays", &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create kernel \"init_rays\": ", err);
    err = clSetKernelArg(init_rays, 0, sizeof(cl_mem), &global.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 0 for kernel \"init_rays\": ", err);
    err = clSetKernelArg(init_rays, 1, sizeof(cl_mem), &ray_list.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 1 for kernel \"init_rays\": ", err);

    CLKernel init_image = clCreateKernel(program, "init_image", &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create kernel \"init_image\": ", err);
    err = clSetKernelArg(init_image, 0, sizeof(cl_mem), &area.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 0 for kernel \"init_image\": ", err);

    CLKernel process = clCreateKernel(program, "process", &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create kernel \"process\": ", err);
    err = clSetKernelArg(process, 0, sizeof(cl_mem), &global.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 0 for kernel \"process\": ", err);
    err = clSetKernelArg(process, 1, sizeof(cl_mem), &area.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 1 for kernel \"process\": ", err);
    err = clSetKernelArg(process, 2, sizeof(cl_mem), &grp_data.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 2 for kernel \"process\": ", err);
    err = clSetKernelArg(process, 3, sizeof(cl_mem), &ray_list.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 3 for kernel \"process\": ", err);
    err = clSetKernelArg(process, 4, sizeof(cl_mem), &grp_list.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 4 for kernel \"process\": ", err);
    err = clSetKernelArg(process, 5, sizeof(cl_mem), &mat_list.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 5 for kernel \"process\": ", err);
    err = clSetKernelArg(process, 6, sizeof(cl_mem), &aabb_list.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 6 for kernel \"process\": ", err);
    err = clSetKernelArg(process, 7, sizeof(cl_mem), &vtx_list.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 7 for kernel \"process\": ", err);
    err = clSetKernelArg(process, 8, sizeof(cl_mem), &tri_list.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 8 for kernel \"process\": ", err);

    CLKernel update_groups = clCreateKernel(program, "update_groups", &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create kernel \"update_groups\": ", err);
    err = clSetKernelArg(update_groups, 0, sizeof(cl_mem), &global.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 0 for kernel \"update_groups\": ", err);
    err = clSetKernelArg(update_groups, 1, sizeof(cl_mem), &grp_data.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 1 for kernel \"update_groups\": ", err);

    CLKernel shuffle_rays = clCreateKernel(program, "shuffle_rays", &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create kernel \"shuffle_rays\": ", err);
    err = clSetKernelArg(shuffle_rays, 0, sizeof(cl_mem), &grp_data.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 0 for kernel \"shuffle_rays\": ", err);
    err = clSetKernelArg(shuffle_rays, 1, sizeof(cl_mem), &ray_list.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 1 for kernel \"shuffle_rays\": ", err);
    //err = clSetKernelArg(shuffle_rays, 2, sizeof(cl_mem), &ray_list.value());
    //if(err != CL_SUCCESS)return opencl_error("Cannot set argument 2 for kernel \"shuffle_rays\": ", err);

    CLKernel update_image = clCreateKernel(program, "update_image", &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create kernel \"update_image\": ", err);
    err = clSetKernelArg(update_image, 0, sizeof(cl_mem), &global.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 0 for kernel \"update_image\": ", err);
    err = clSetKernelArg(update_image, 1, sizeof(cl_mem), &area.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 1 for kernel \"update_image\": ", err);
    err = clSetKernelArg(update_image, 2, sizeof(cl_mem), &image.value());
    if(err != CL_SUCCESS)return opencl_error("Cannot set argument 2 for kernel \"update_image\": ", err);


    err = clEnqueueNDRangeKernel(queue, init_image, 1, 0, &area_size, &unit_width, 0, 0, 0);
    if(err != CL_SUCCESS)return opencl_error("Cannot execute \"init_image\": ", err);

    glFinish();
    err = clEnqueueAcquireGLObjects(queue, 1, &image.value(), 0, 0, 0);
    if(err != CL_SUCCESS)return opencl_error("Cannot acquire \"image\" from OpenGL: ", err);

    err = clEnqueueNDRangeKernel(queue, update_image, 1, 0, &area_size, &unit_width, 0, 0, 0);
    if(err != CL_SUCCESS)return opencl_error("Cannot execute \"update_image\": ", err);

    err = clEnqueueReleaseGLObjects(queue, 1, &image.value(), 0, 0, 0);
    if(err != CL_SUCCESS)return opencl_error("Cannot release \"image\" to OpenGL: ", err);
    glFinish();


    // Main loop

    for(SDL_Event evt;;)
    {
        SDL_WaitEvent(&evt);  if(evt.type == SDL_QUIT)break;

        glBegin(GL_TRIANGLE_STRIP);
        glTexCoord2f(0, 0);  glVertex3f(-1, -1, 0);
        glTexCoord2f(0, 1);  glVertex3f(-1, +1, 0);
        glTexCoord2f(1, 0);  glVertex3f(+1, -1, 0);
        glTexCoord2f(1, 1);  glVertex3f(+1, +1, 0);
        glEnd();

        SDL_GL_SwapBuffers();  //SDL_Delay(10);
    }
    return 0;
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

    if(SDL_Init(SDL_INIT_VIDEO))return sdl_error("SDL_Init failed: ");
    int res = ray_tracer(platform[index], device);  SDL_Quit();  return res;
}
