// main.cpp -- entry point
//

#include "model.h"
#include "cl-helper.h"
#include <SDL/SDL.h>
#include <SDL/SDL_opengl.h>
#include <GL/glx.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>

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


    size_t warp_width, unit_width, width, height, area_size, ray_count, group_count;
    GLTexture texture;  CLContext context;  cl_device_id device;  CLQueue queue;  CLProgram program;
    CLBuffer global, area, ray_list, grp_data, ray_index[2], grp_list, mat_list, aabb_list, vtx_list, tri_list, image;
    Kernel init_groups, init_rays, init_image, process, count_groups, update_groups, set_ray_index, update_image;

    size_t sort_block, block_count;
    CLBuffer sort_count, local_index, global_index;
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

    bool create_sub_buffer(CLBuffer &buf, const char *name, cl_mem from, cl_mem_flags flags, size_t offs, size_t size)
    {
        cl_buffer_region region = {offs, size};  cl_int err;
        buf = clCreateSubBuffer(from, flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);  if(err == CL_SUCCESS)return true;
        cout << "Cannot create sub-buffer \"" << name << "\": " << cl_error_string(err) << endl;  return false;
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
        printf("Global data: %X %X %X\n", data.group_count, data.pixel_offset, data.ray_count);

        const size_t n = 4;  GroupData buf[n];
        err = clEnqueueReadBuffer(queue, grp_data, CL_TRUE, 0, sizeof(buf), buf, 0, 0, 0);
        if(err != CL_SUCCESS)return opencl_error("Cannot read buffer data: ", err);
        for(size_t i = 0; i < n; i++)
            printf("%8X %8X %8X %8X %8X %8X\n", buf[i].base.s[0], buf[i].base.s[1],
                buf[i].count.s[0], buf[i].count.s[1], buf[i].offset.s[0], buf[i].offset.s[1]);
        printf("   -    -    -    -    -    -    -    -    -    -    -    -   \n");
        err = clEnqueueReadBuffer(queue, grp_data, CL_TRUE, (group_count - n) * sizeof(GroupData), sizeof(buf), buf, 0, 0, 0);
        if(err != CL_SUCCESS)return opencl_error("Cannot read buffer data: ", err);
        for(size_t i = 0; i < n; i++)
            printf("%8X %8X %8X %8X %8X %8X\n", buf[i].base.s[0], buf[i].base.s[1],
                buf[i].count.s[0], buf[i].count.s[1], buf[i].offset.s[0], buf[i].offset.s[1]);
        printf("--------------------------------------------------------------\n");
        return data.ray_count <= ray_count && !(data.ray_count % warp_width);
    }

    bool check_sorting(cl_uint mask)  // DEBUG
    {
        cl_uint2 *buf = new cl_uint2[ray_count];
        cl_int err = clEnqueueReadBuffer(queue, ray_index[0], CL_TRUE, 0, ray_count * sizeof(cl_uint2), buf, 0, 0, 0);
        if(err != CL_SUCCESS)
        {
            delete [] buf;  return opencl_error("Cannot read buffer data: ", err);
        }
        int limit = 8;
        for(size_t i = 1; i < ray_count; i++)if((buf[i].s[0] & mask) < (buf[i - 1].s[0] & mask))
        {
            printf("Invalid for index %8zX: %08X > %08X (mask %08X)!!!\n", i, buf[i - 1].s[0], buf[i].s[0], mask);
            if(!--limit)break;
        }
        delete [] buf;  return true;
    }


    bool init_gl();
    bool init_cl(cl_platform_id platform);
    bool build_program();
    bool create_buffers();
    bool create_buffers(GlobalData &data, Group *grp, Matrix *mat, size_t mat_count,
        AABB *aabb, size_t aabb_count, Vertex *vtx, size_t vtx_count, cl_uint *tri, size_t tri_count);
    bool create_kernels();

    static size_t align(size_t val, size_t unit)
    {
        return (val + unit - 1) / unit * unit;
    }

public:
    RayTracer(size_t width_, size_t height_, size_t ray_count_) :
        warp_width(32), unit_width(512), width(width_), height(height_), area_size(width_ * height_), sort_block(16)
    {
        ray_count = align(ray_count_, unit_width * sort_block);
        block_count = ray_count / (unit_width * sort_block);
    }

    bool init(cl_platform_id platform)
    {
        return init_gl() && init_cl(platform) && build_program() && create_buffers() && create_kernels();
    }

    bool init_frame();
    bool make_step();
    bool draw_frame();

    cl_uint current_ray()
    {
        GlobalData data;
        cl_int err = clEnqueueReadBuffer(queue, global, CL_TRUE, 0, sizeof(data), &data, 0, 0, 0);
        if(err == CL_SUCCESS)return data.pixel_offset;  opencl_error("Cannot read buffer data: ", err);  return 0;
    }
};


bool RayTracer::init_gl()
{
    glGenTextures(1, &texture.value());  glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glEnable(GL_TEXTURE_2D);  glColor3f(1, 1, 1);  return true;
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
    sprintf(buf, "-DWARP_WIDTH=%zu -DUNIT_WIDTH=%zu -DSORT_BLOCK=%zu "
        "-cl-mad-enable -cl-nv-verbose", warp_width, unit_width, sort_block);
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


bool RayTracer::create_buffers()
{
    const size_t tri_threshold = 128, aabb_threshold = 128;

    const size_t n_obj = 256;
    Matrix mat[n_obj];  memset(mat, 0, sizeof(mat));
    for(size_t i = 0; i < n_obj; i++)
    {
        double alpha = 2 * 3.14159265359 * random() / RAND_MAX;
        mat[i].x.s[0] = mat[i].y.s[2] = cos(alpha);
        mat[i].x.s[2] = -(mat[i].y.s[0] = sin(alpha));
        mat[i].z.s[1] = 1;

        mat[i].x.s[3] = 4.0 * random() / RAND_MAX - 2;
        mat[i].y.s[3] = 4.0 * random() / RAND_MAX;
        mat[i].z.s[3] = 2.0 * random() / RAND_MAX - 1;
    }
    ResourceManager mngr;  mngr.reserve_groups(6);
    mngr.reserve_aabbs(n_obj);


    Model bunny;
    cout << "Loading bunny model..." << endl;
    if(!bunny.load("bun_zipper.ply"))
    {
        cout << "Failed to load bunny model!" << endl;  return false;
    }
    bunny.subdivide(tri_threshold, aabb_threshold);
    bunny.reserve(mngr);


    Model dragon;
    cout << "Loading dragon model..." << endl;
    if(!dragon.load("dragon_vrip.ply"))
    //if(!dragon.load("bun_zipper.ply"))
    {
        cout << "Failed to load dragon model!" << endl;  return false;
    }
    dragon.subdivide(tri_threshold, aabb_threshold);
    dragon.reserve(mngr);


    mngr.alloc();  mngr.get_groups(3);  // predefined (spawn, sky, light)
    cl_uint green_id = make_group_id(mngr.get_groups(1), tr_none, sh_material);
    cl_uint red_id = make_group_id(mngr.get_groups(1), tr_none, sh_material);
    cl_uint aabb_id = make_group_id(mngr.get_groups(1), tr_identity, sh_aabb);

    Group *grp = mngr.group(aabb_id & GROUP_ID_MASK);
    AABB *aabb = mngr.aabb(grp->aabb.aabb_offs = mngr.get_aabbs(n_obj));
    grp->aabb.aabb_count = n_obj;  grp->aabb.flags = f_local0;

    grp = mngr.group(green_id & GROUP_ID_MASK);  grp->material.color.s[3] = 0.1;
    grp->material.color.s[0] = 0.2;  grp->material.color.s[1] = 0.9;  grp->material.color.s[2] = 0.2;

    grp = mngr.group(red_id & GROUP_ID_MASK);  grp->material.color.s[3] = 0.1;
    grp->material.color.s[0] = 0.9;  grp->material.color.s[1] = 0.2;  grp->material.color.s[2] = 0.2;

    bunny.fill(mngr, green_id);  dragon.fill(mngr, red_id);
    for(size_t i = 0; i < n_obj; i++)(i & 1 ? dragon : bunny).put(aabb[i], mat[i], i);
    assert(mngr.full());


    GlobalData data;  data.ray_count = ray_count;
    data.group_count = group_count = align(mngr.group_count() + 1, unit_width);
    cout << "Group count: " << group_count << endl;

    data.cam.eye.s[0] = 0;  data.cam.eye.s[1] = -0.3;  data.cam.eye.s[2] = 0;
    data.cam.top_left.s[0] = -0.5;  data.cam.top_left.s[1] = 1;  data.cam.top_left.s[2] = -0.5;
    data.cam.dx.s[0] = 1.0 / width;  data.cam.dx.s[1] = 0;  data.cam.dx.s[2] = 0;
    data.cam.dy.s[0] = 0;  data.cam.dy.s[1] = 0;  data.cam.dy.s[2] = 1.0 / height;
    data.cam.width = width;  data.cam.height = height;
    data.cam.root_group = aabb_id;  data.cam.root_local = 0;

    return create_buffers(data, mngr.group(0), mat, n_obj, mngr.aabb(0), mngr.aabb_count(),
        mngr.vertex(0), mngr.vertex_count(), mngr.triangle(0), mngr.triangle_count());
}

bool RayTracer::create_buffers(GlobalData &data, Group *grp, Matrix *mat, size_t mat_count,
    AABB *aabb, size_t aabb_count, Vertex *vtx, size_t vtx_count, cl_uint *tri, size_t tri_count)
{
    if(!create_buffer(global, "global", mem_copy, sizeof(data), &data))return false;
    if(!create_buffer(area, "area", mem_rw, area_size * sizeof(cl_float4)))return false;
    if(!create_buffer(ray_list, "ray_list", mem_rw, ray_count * sizeof(RayQueue)))return false;
    if(!create_buffer(grp_data, "grp_data", mem_rw, data.group_count * sizeof(GroupData)))return false;
    if(!create_buffer(ray_index[0], "ray_index[0]", mem_rw, ray_count * sizeof(cl_uint2)))return false;
    if(!create_buffer(ray_index[1], "ray_index[1]", mem_rw, ray_count * sizeof(cl_uint2)))return false;
    if(!create_buffer(grp_list, "grp_list", mem_ro | mem_copy, group_count * sizeof(Group), grp))return false;
    if(!create_buffer(mat_list, "mat_list", mem_ro | mem_copy, mat_count * sizeof(Matrix), mat))return false;
    if(!create_buffer(aabb_list, "aabb_list", mem_ro | mem_copy, aabb_count * sizeof(AABB), aabb))return false;
    if(!create_buffer(vtx_list, "vtx_list", mem_ro | mem_copy, vtx_count * sizeof(Vertex), vtx))return false;
    if(!create_buffer(tri_list, "tri_list", mem_ro | mem_copy, tri_count * sizeof(cl_uint), tri))return false;

    cl_int err;
    image = clCreateFromGLTexture2D(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texture, &err);
    if(err != CL_SUCCESS)return opencl_error("Cannot create image: ", err);

    // sort

    if(!create_sub_buffer(sort_count, "sort_count", global, mem_rw, offsetof(GlobalData, ray_count), sizeof(cl_uint)))return false;
    if(!create_buffer(local_index, "local_index", mem_rw, ray_count * sizeof(cl_uint)))return false;
    if(!create_buffer(global_index, "global_index", mem_rw, block_count * RADIX_MAX * sizeof(cl_uint)))return false;
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
    if(!set_kernel_arg(set_ray_index, 0, global))return false;
    if(!set_kernel_arg(set_ray_index, 1, grp_data))return false;

    if(!create_kernel(update_image, "update_image"))return false;
    if(!set_kernel_arg(update_image, 0, global))return false;
    if(!set_kernel_arg(update_image, 1, area))return false;
    if(!set_kernel_arg(update_image, 2, image))return false;

    // sort

    if(!create_kernel(local_count, "local_count"))return false;
    if(!set_kernel_arg(local_count, 1, sort_count))return false;
    if(!set_kernel_arg(local_count, 2, local_index))return false;
    if(!set_kernel_arg(local_count, 3, global_index))return false;

    if(!create_kernel(global_count, "global_count"))return false;
    if(!set_kernel_arg(global_count, 0, global_index))return false;
    if(!set_kernel_arg(global_count, 1, sort_count))return false;

    if(!create_kernel(shuffle_data, "shuffle_data"))return false;
    if(!set_kernel_arg(shuffle_data, 2, sort_count))return false;
    if(!set_kernel_arg(shuffle_data, 3, local_index))return false;
    if(!set_kernel_arg(shuffle_data, 4, global_index))return false;
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

    for(cl_uint shift = 0, mask = GROUP_ID_MASK, max = group_count - 1; max;
        shift += RADIX_SHIFT, mask >>= RADIX_SHIFT, max >>= RADIX_SHIFT)
    {
        if(!set_kernel_arg(local_count, 0, ray_index[0]))return false;
        if(!set_kernel_arg(local_count, 4, shift))return false;
        if(!set_kernel_arg(local_count, 5, mask & RADIX_MASK))return false;
        if(!set_kernel_arg(local_count, 6, std::min(cl_uint(RADIX_MAX), max + 1)))return false;
        if(!run_kernel(local_count, block_count * unit_width))return false;

        if(!set_kernel_arg(global_count, 2, std::min(cl_uint(RADIX_MAX), max + 1)))return false;
        if(!run_kernel(global_count, unit_width))return false;

        if(!set_kernel_arg(shuffle_data, 0, ray_index[0]))return false;
        if(!set_kernel_arg(shuffle_data, 1, ray_index[1]))return false;
        if(!set_kernel_arg(shuffle_data, 5, shift))return false;
        if(!set_kernel_arg(shuffle_data, 6, mask & RADIX_MASK))return false;
        if(!set_kernel_arg(shuffle_data, 7, max < (1 << RADIX_SHIFT)))return false;
        if(!run_kernel(shuffle_data, block_count * unit_width))return false;
        swap(ray_index[0].value(), ray_index[1].value());
    }
    if(!set_kernel_arg(count_groups, 2, ray_index[0]))return false;
    if(!run_kernel(count_groups, ray_count))return false;

    if(!run_kernel(update_groups, unit_width))return false;

    if(!set_kernel_arg(set_ray_index, 2, ray_index[0]))return false;
    if(!set_kernel_arg(set_ray_index, 3, ray_index[1]))return false;
    if(!run_kernel(set_ray_index, ray_count))return false;
    swap(ray_index[0].value(), ray_index[1].value());

    //if(!debug_print())return false;  // DEBUG
    //if(!check_sorting(GROUP_ID_MASK))return false;  // DEBUG
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

    const int width = 1024, height = 1024;
    SDL_Surface *surface = SDL_SetVideoMode(width, height, 0, SDL_HWSURFACE | SDL_OPENGL);
    if(!surface)return sdl_error("Cannot create OpenGL context: ");
    SDL_WM_SetCaption("RayTracer 1.0", 0);

    const int repeat_count = 32;
    RayTracer ray_tracer(width, height, 1024 * 1024);
    if(!ray_tracer.init(platform))return false;
    glViewport(0, 0, width, height);
    cout << "Ready." << endl;

    if(!ray_tracer.init_frame())return false;
    if(!ray_tracer.draw_frame())return false;

    cl_uint cur_ray = 0;
    cout << setprecision(3) << fixed;
    for(SDL_Event evt;;)
    {
        SDL_WaitEvent(&evt);
        switch(evt.type)
        {
        case SDL_QUIT:  return true;
        case SDL_MOUSEBUTTONDOWN:
            {
                nsec_type start = get_time();  cl_uint old_ray = cur_ray;
                for(int i = 0; i < repeat_count; i++)if(!ray_tracer.make_step())return false;
                if(!ray_tracer.draw_frame())return false;

                double delta = (get_time() - start) * 1e-9;  cur_ray = ray_tracer.current_ray();
                cout << "Frame ready in " << delta << " s, " << (cur_ray - old_ray) << " rays, " <<
                    1e-6 * (cur_ray - old_ray) / delta << " MR/s."<< endl;
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
