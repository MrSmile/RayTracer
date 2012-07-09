// main.cpp -- entry point
//

#include "cl-helper.h"
#include <SDL/SDL.h>
#include <SDL/SDL_opengl.h>
#include <GL/glx.h>
#include <iostream>
#include <iomanip>
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



typedef long long nsec_type;

inline nsec_type get_time()
{
    timespec ts;  clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000000000 * nsec_type(ts.tv_sec) + ts.tv_nsec;
}



inline cl_int delete_texture(GLuint tex)
{
    glDeleteTextures(1, &tex);  return 0;
}

//typedef AutoReleaser<SDL_Window *, SDL_DestroyWindow> SDLWindow;
//typedef AutoReleaser<SDL_GL_Context, SDL_DeleteContext> GLContext;
typedef AutoReleaser<GLuint, delete_texture> GLTexture;



bool sdl_error(const char *text)
{
    cout << text << SDL_GetError() << endl;  SDL_ClearError();  return false;
}

bool opencl_error(const char *text, cl_int err)
{
    cout << text << cl_error_string(err) << endl;  return false;
}


class RayTracer
{
    struct Kernel : public CLKernel
    {
        const char *name;

        Kernel() : name(0)
        {
        }

        cl_kernel operator = (cl_kernel kernel)
        {
            return attach(kernel);
        }
    };


    static const int sort_pass_count = 16 / RADIX_SHIFT;

    size_t unit_width, width, height, area_size, ray_count, group_count;  int flip;
    GLTexture texture;  CLContext context;  cl_device_id device;  CLQueue queue;  CLProgram program;
    CLBuffer global, area, ray_list, grp_data, ray_index[2], grp_list, mat_list, aabb_list, vtx_list, tri_list, image;
    Kernel init_groups, init_rays, init_image, process, count_groups, update_groups, set_ray_index, update_image;

    CLBuffer local_index, global_index;
    Kernel local_count, global_count, shuffle_data;


    enum BufferFlags
    {
        mem_rw    = CL_MEM_READ_WRITE,
        mem_wo    = CL_MEM_WRITE_ONLY,
        mem_ro    = CL_MEM_READ_ONLY,
        mem_use   = CL_MEM_USE_HOST_PTR,
        mem_alloc = CL_MEM_ALLOC_HOST_PTR,
        mem_copy  = CL_MEM_COPY_HOST_PTR
    };

    bool create_buffer(CLBuffer &buf, const char *name, cl_mem_flags flags, size_t size, void *ptr = 0)
    {
        cl_int err;  buf = clCreateBuffer(context, flags, size, ptr, &err);  if(err == CL_SUCCESS)return true;
        cout << "Cannot create buffer \"" << name << "\": " << cl_error_string(err) << endl;  return false;
    }

    bool create_kernel(Kernel &kernel, const char *name)
    {
        cl_int err;  kernel = clCreateKernel(program, kernel.name = name, &err);  if(err == CL_SUCCESS)return true;
        cout << "Cannot create kernel \"" << name << "\": " << cl_error_string(err) << endl;  return false;
    }

    bool set_kernel_arg(const Kernel &kernel, cl_uint arg, size_t size, const void *ptr)
    {
        cl_int err = clSetKernelArg(kernel, arg, size, ptr);  if(err == CL_SUCCESS)return true;
        cout << "Cannot set argument " << arg << " for kernel \"" <<
            kernel.name << "\": " << cl_error_string(err) << endl;  return false;
    }

    bool set_kernel_arg(const Kernel &kernel, cl_uint arg, cl_mem buf)
    {
        return set_kernel_arg(kernel, arg, sizeof(cl_mem), &buf);
    }

    bool set_kernel_arg(const Kernel &kernel, cl_uint arg, cl_uint val)
    {
        return set_kernel_arg(kernel, arg, sizeof(cl_uint), &val);
    }

    bool run_kernel(const Kernel &kernel, size_t size)
    {
        cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &size, &unit_width, 0, 0, 0);  if(err == CL_SUCCESS)return true;
        cout << "Cannot execute kernel \"" << kernel.name << "\": " << cl_error_string(err) << endl;  return false;
    }


    bool debug_print()  // DEBUG
    {
        GlobalData data;
        cl_int err = clEnqueueReadBuffer(queue, global, CL_TRUE, 0, sizeof(data), &data, 0, 0, 0);
        if(err != CL_SUCCESS)return opencl_error("Cannot read buffer data: ", err);

        const size_t n = 8;  GroupData buf[n];
        err = clEnqueueReadBuffer(queue, grp_data, CL_TRUE, 0, sizeof(buf), buf, 0, 0, 0);
        if(err != CL_SUCCESS)return opencl_error("Cannot read buffer data: ", err);

        printf("Global data: %X %X %X\n", data.group_count, data.pixel_offset, data.ray_count);
        for(size_t i = 0; i < n; i++)
            printf("%8X %8X %8X %8X\n", buf[i].count.s[0], buf[i].count.s[1], buf[i].offset.s[0], buf[i].offset.s[1]);
        printf("------------------------\n");  return true;
    }


    bool init_gl();
    bool init_cl(cl_platform_id platform);
    bool build_program();
    bool create_buffers();
    bool create_kernels();

    static size_t align(size_t val, size_t unit)
    {
        return (val + unit - 1) / unit * unit;
    }

public:
    RayTracer(size_t width_, size_t height_, size_t ray_count_) :
        unit_width(256), width(width_), height(height_), area_size(width_ * height_), ray_count(ray_count_), flip(0)
    {
        ray_count = align(ray_count_, unit_width * SORT_BLOCK);
    }

    bool init(cl_platform_id platform)
    {
        return init_gl() && init_cl(platform) && build_program() && create_buffers() && create_kernels();
    }

    bool init_frame();
    bool make_step();
    bool draw_frame();
};


bool RayTracer::init_gl()
{
    glGenTextures(1, &texture.value());  glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glEnable(GL_TEXTURE_2D);  glColor3f(1, 1, 1);  glViewport(0, 0, width, height);
    return true;
}

bool RayTracer::init_cl(cl_platform_id platform)
{
    cl_context_properties prop[] =
    {
        CL_GL_CONTEXT_KHR, cl_context_properties(glXGetCurrentContext()),
        CL_GLX_DISPLAY_KHR, cl_context_properties(glXGetCurrentDisplay()),
        CL_CONTEXT_PLATFORM, cl_context_properties(platform), 0
    };

    cl_int err;
    context = clCreateContextFromType(prop, CL_DEVICE_TYPE_GPU, 0, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create context: ", err);

    size_t res_size;
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device), &device, &res_size);
    if(err != CL_SUCCESS)return opencl_error("Cannot get device from context: ", err);
    if(res_size < sizeof(device))
    {
        cout << "Cannot get device from context!" << endl;  return false;
    }

    queue = clCreateCommandQueue(context, device, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create command queue: ", err);
    return true;
}

bool RayTracer::build_program()
{
    const char *src = "#include \"ray-tracer.cl\"";  cl_int err;
    program = clCreateProgramWithSource(context, 1, &src, 0, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create program: ", err);

    char buf[65536];
    sprintf(buf, "-DUNIT_WIDTH=%zu -cl-nv-verbose -w", unit_width);
    int build_err = clBuildProgram(program, 1, &device, buf, 0, 0);
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buf), buf, 0);
    if(err != CL_SUCCESS)return opencl_error("Cannot get build info: ", err);
    // The OpenCL Specification, version 1.1, revision 44 (6/1/11), section 4.1, page 33 footnote:
    // A null terminated string is returned by OpenCL query function calls if the return type of the information being
    // queried is a char[].

    cout << "Build log:\n" << buf << endl;
    if(build_err == CL_SUCCESS)cout << "Compilation successfull." << endl;
    else cout << "Compilation failed: " << cl_error_string(build_err) << endl;
    return build_err == CL_SUCCESS;
}


inline cl_uint make_group_id(int index, int transform, int shader)
{
    return index | transform << GROUP_TR_SHIFT | shader << GROUP_SH_SHIFT;
}

bool RayTracer::create_buffers()
{
    const float pi = 3.14159265358979323846264338327950288;
    const int N = 64;  Vertex vtx[2 * N];  cl_uint tri[2 * N];
    for(int i = 0; i < N; i++)
    {
        cl_uint dn = 2 * i, up = dn + 1;
        vtx[dn].norm.s[0] = vtx[up].norm.s[0] = vtx[dn].pos.s[0] = vtx[up].pos.s[0] = cos(i * (2 * pi / N));
        vtx[dn].norm.s[1] = vtx[up].norm.s[1] = vtx[dn].pos.s[1] = vtx[up].pos.s[1] = sin(i * (2 * pi / N));
        vtx[dn].norm.s[2] = vtx[up].norm.s[2] = 0;  vtx[dn].pos.s[2] = -0.5f;  vtx[up].pos.s[2] = 0.5f;

        cl_uint dn1 = (i + 1) % N * 2, up1 = dn1 + 1;
        tri[dn] = dn | up << 10 | dn1 << 20;  tri[up] = up1 | dn1 << 10 | up << 20;
    }

    /*vtx[0].pos.s[0] = -1;  vtx[0].pos.s[1] = 0;  vtx[0].pos.s[2] = -1;
    vtx[1].pos.s[0] = +1;  vtx[1].pos.s[1] = 0;  vtx[1].pos.s[2] = -1;
    vtx[2].pos.s[0] = -1;  vtx[2].pos.s[1] = 0;  vtx[2].pos.s[2] = +1;
    vtx[0].norm.s[0] = vtx[1].norm.s[0] = vtx[2].norm.s[0] = 0;
    vtx[0].norm.s[1] = vtx[1].norm.s[1] = vtx[2].norm.s[1] = -1;
    vtx[0].norm.s[2] = vtx[1].norm.s[2] = vtx[2].norm.s[2] = 0;
    tri[0] = 0 | 1 << 10 | 2 << 20;*/

    const int n_group = 4;  cl_uint group_id[n_group];
    Group grp[n_group];  group_count = align(n_group, unit_width);

    int index;
    group_id[0] = make_group_id(0, tr_none, sh_spawn);
    group_id[1] = make_group_id(1, tr_none, sh_sky);

    index = 2;
    group_id[index] = make_group_id(index, tr_none, sh_material);

    index = 3;
    group_id[index] = make_group_id(index, tr_identity, sh_mesh);
    grp[index].mesh.vtx_offs = 0;
    grp[index].mesh.tri_offs = 0;
    grp[index].mesh.tri_count = 2 * N;
    grp[index].mesh.material_id = group_id[2];


    GlobalData data;
    data.group_count = group_count;  data.ray_count = ray_count;

    data.cam.eye.s[0] = 0;  data.cam.eye.s[1] = -5;  data.cam.eye.s[2] = 0;
    data.cam.top_left.s[0] = -0.5;  data.cam.top_left.s[1] = 1;  data.cam.top_left.s[2] = -0.5;
    data.cam.dx.s[0] = 1.0 / width;  data.cam.dx.s[1] = 0;  data.cam.dx.s[2] = 0;
    data.cam.dy.s[0] = 0;  data.cam.dy.s[1] = 0;  data.cam.dy.s[2] = 1.0 / height;
    data.cam.width = width;  data.cam.height = height;
    data.cam.root_group = group_id[3];  data.cam.root_local = 0;

    if(!create_buffer(global, "global", mem_copy, sizeof(data), &data))return false;
    if(!create_buffer(area, "area", mem_rw, area_size * sizeof(cl_float4)))return false;
    if(!create_buffer(ray_list, "ray_list", mem_rw, ray_count * sizeof(RayQueue)))return false;
    if(!create_buffer(grp_data, "grp_data", mem_rw, data.group_count * sizeof(GroupData)))return false;
    if(!create_buffer(ray_index[0], "ray_index[0]", mem_rw, ray_count * sizeof(cl_uint2)))return false;
    if(!create_buffer(ray_index[1], "ray_index[1]", mem_rw, ray_count * sizeof(cl_uint2)))return false;
    if(!create_buffer(grp_list, "grp_list", mem_ro | mem_copy, sizeof(grp), grp))return false;
    if(!create_buffer(mat_list, "mat_list", mem_ro, 1))return false;  // TODO
    if(!create_buffer(aabb_list, "aabb_list", mem_ro, 1))return false;  // TODO
    if(!create_buffer(vtx_list, "vtx_list", mem_ro | mem_copy, sizeof(vtx), vtx))return false;
    if(!create_buffer(tri_list, "tri_list", mem_ro | mem_copy, sizeof(tri), tri))return false;

    cl_int err;
    image = clCreateFromGLTexture2D(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texture, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create image: ", err);

    // sort

    if(!create_buffer(local_index, "local_index", mem_rw, ray_count * sizeof(cl_uint)))return false;
    if(!create_buffer(global_index, "global_index", mem_rw, (ray_count / SORT_BLOCK) * sizeof(cl_uint)))return false;
    return true;
}

bool RayTracer::create_kernels()
{
    if(!create_kernel(init_groups, "init_groups"))return false;
    if(!set_kernel_arg(init_groups, 0, grp_data))return false;

    if(!create_kernel(init_rays, "init_rays"))return false;
    if(!set_kernel_arg(init_rays, 0, global))return false;
    if(!set_kernel_arg(init_rays, 1, ray_list))return false;
    if(!set_kernel_arg(init_rays, 2, ray_index[0]))return false;

    if(!create_kernel(init_image, "init_image"))return false;
    if(!set_kernel_arg(init_image, 0, area))return false;

    if(!create_kernel(process, "process"))return false;
    if(!set_kernel_arg(process, 0, global))return false;
    if(!set_kernel_arg(process, 1, area))return false;
    if(!set_kernel_arg(process, 2, ray_list))return false;
    if(!set_kernel_arg(process, 4, grp_list))return false;
    if(!set_kernel_arg(process, 5, mat_list))return false;
    if(!set_kernel_arg(process, 6, aabb_list))return false;
    if(!set_kernel_arg(process, 7, vtx_list))return false;
    if(!set_kernel_arg(process, 8, tri_list))return false;

    if(!create_kernel(count_groups, "count_groups"))return false;
    if(!set_kernel_arg(count_groups, 0, global))return false;
    if(!set_kernel_arg(count_groups, 1, grp_data))return false;

    if(!create_kernel(update_groups, "update_groups"))return false;
    if(!set_kernel_arg(update_groups, 0, global))return false;
    if(!set_kernel_arg(update_groups, 1, grp_data))return false;

    if(!create_kernel(set_ray_index, "set_ray_index"))return false;
    if(!set_kernel_arg(set_ray_index, 0, grp_data))return false;

    if(!create_kernel(update_image, "update_image"))return false;
    if(!set_kernel_arg(update_image, 0, global))return false;
    if(!set_kernel_arg(update_image, 1, area))return false;
    if(!set_kernel_arg(update_image, 2, image))return false;

    // sort

    if(!create_kernel(local_count, "local_count"))return false;
    if(!set_kernel_arg(local_count, 1, local_index))return false;
    if(!set_kernel_arg(local_count, 2, global_index))return false;

    if(!create_kernel(global_count, "global_count"))return false;
    if(!set_kernel_arg(global_count, 0, global_index))return false;
    if(!set_kernel_arg(global_count, 1, ray_count / (unit_width * SORT_BLOCK)))return false;

    if(!create_kernel(shuffle_data, "shuffle_data"))return false;
    if(!set_kernel_arg(shuffle_data, 2, local_index))return false;
    if(!set_kernel_arg(shuffle_data, 3, global_index))return false;
    return true;
}


bool RayTracer::init_frame()
{
    if(!run_kernel(init_groups, group_count))return false;
    if(!run_kernel(init_rays, ray_count))return false;
    if(!run_kernel(init_image, area_size))return false;
    return true;
}

bool RayTracer::make_step()
{
    if(!set_kernel_arg(process, 3, ray_index[0]))return false;
    if(!run_kernel(process, ray_count))return false;
    for(int i = 0; i < sort_pass_count; i++)
    {
        if(!set_kernel_arg(local_count, 0, ray_index[0]))return false;
        if(!set_kernel_arg(local_count, 3, i))return false;
        if(!run_kernel(local_count, ray_count / SORT_BLOCK))return false;
        if(!run_kernel(global_count, unit_width))return false;
        if(!set_kernel_arg(shuffle_data, 0, ray_index[0]))return false;
        if(!set_kernel_arg(shuffle_data, 1, ray_index[1]))return false;
        if(!set_kernel_arg(shuffle_data, 4, i))return false;
        if(!run_kernel(shuffle_data, ray_count / SORT_BLOCK))return false;
        swap(ray_index[0].value(), ray_index[1].value());
    }
    if(!set_kernel_arg(count_groups, 2, ray_index[0]))return false;
    if(!run_kernel(count_groups, ray_count))return false;
    if(!run_kernel(update_groups, unit_width))return false;
    if(!set_kernel_arg(set_ray_index, 1, ray_index[0]))return false;
    if(!set_kernel_arg(set_ray_index, 2, ray_index[1]))return false;
    if(!run_kernel(set_ray_index, ray_count))return false;
    swap(ray_index[0].value(), ray_index[1].value());
    //if(!debug_print())return false;  // DEBUG
    return true;
}

bool RayTracer::draw_frame()
{
    glFinish();
    cl_int err = clEnqueueAcquireGLObjects(queue, 1, &image.value(), 0, 0, 0);
    if(err != CL_SUCCESS)return opencl_error("Cannot acquire image from OpenGL: ", err);

    if(!run_kernel(update_image, area_size))return false;

    err = clEnqueueReleaseGLObjects(queue, 1, &image.value(), 0, 0, 0);
    if(err != CL_SUCCESS)return opencl_error("Cannot release image to OpenGL: ", err);
    glFinish();  return true;
}



bool ray_tracer(cl_platform_id platform)
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

    const int repeat_count = 32;
    RayTracer ray_tracer(width, height, 256 * 1024);
    if(!ray_tracer.init(platform))return false;

    if(!ray_tracer.init_frame())return false;
    if(!ray_tracer.draw_frame())return false;

    cout << setprecision(3) << fixed;
    for(SDL_Event evt;;)
    {
        SDL_WaitEvent(&evt);
        switch(evt.type)
        {
        case SDL_QUIT:  return true;
        case SDL_MOUSEBUTTONDOWN:
            {
                nsec_type start = get_time();
                for(int i = 0; i < repeat_count; i++)if(!ray_tracer.make_step())return false;
                if(!ray_tracer.draw_frame())return false;
                cout << "Frame ready in " << (get_time() - start) * 1e-9 << "s." << endl;
            }
        case SDL_VIDEOEXPOSE:  break;
        default:  continue;
        }

        glBegin(GL_TRIANGLE_STRIP);
        glTexCoord2f(0, 0);  glVertex3f(-1, -1, 0);
        glTexCoord2f(0, 1);  glVertex3f(-1, +1, 0);
        glTexCoord2f(1, 0);  glVertex3f(+1, -1, 0);
        glTexCoord2f(1, 1);  glVertex3f(+1, +1, 0);
        glEnd();  SDL_GL_SwapBuffers();
    }
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
        cout << "Invalid platform index!" << endl;  return -1;
    }

    if(SDL_Init(SDL_INIT_VIDEO))return sdl_error("SDL_Init failed: ");
    int res = ray_tracer(platform[index]) ? 0 : -1;
    SDL_Quit();  return res;
}
