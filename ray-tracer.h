// ray-tracer.h : OpenCL data layout
//

#ifdef __cplusplus
#define uint    cl_uint
#define uint2   cl_uint2
#define float   cl_float
#define float3  cl_float3
#define float4  cl_float4
#endif


// material shader

typedef struct
{
    float4 color;  // color.w -- specular intensity
} MatShader;


// AABB shader

typedef union
{
    struct
    {
        float3 min, max;
    };
    struct
    {
        uint res1_[3], group_id;
        uint res2_[3], local_id;
    };
} AABB;

enum AABBFlags
{
    f_local0 = 1, f_local1 = 2
};

typedef struct
{
    uint aabb_offs, aabb_count, flags;
} AABBShader;


// mesh shader

typedef struct
{
    float3 pos, norm;
} Vertex;

typedef struct
{
    uint vtx_offs, tri_offs, tri_count, material_id;
} MeshShader;


// group description

#define GROUP_ID_MASK  0xFFFFFF
#define GROUP_TR_SHIFT       24
#define GROUP_TR_MASK       0xF
#define GROUP_SH_SHIFT       28
#define GROUP_SH_MASK       0xF

enum TransformType
{
    tr_none = 0, tr_identity, tr_ortho, tr_affine
};

enum ShaderType
{
    sh_spawn = 0, sh_sky, sh_light, sh_material, sh_aabb, sh_mesh
};

enum PredefinedGroups
{
    spawn_group = 0,
    sky_group = 1 | sh_sky << GROUP_SH_SHIFT,
    light_group = 2 | sh_light << GROUP_SH_SHIFT
};

typedef struct
{
    float4 x, y, z;
} Matrix;

typedef union
{
    MatShader material;
    AABBShader aabb;
    MeshShader mesh;
} Group;


// global scene data

typedef struct
{
    float3 eye, top_left, dx, dy;
    uint width, height, root_group, root_local;
} Camera;

typedef struct
{
    uint pixel_offset, pixel_count;
    uint group_count, old_count, ray_count;  // counts must be multiple of UNIT_WIDTH
    Camera cam;
} GlobalData;


// internal data structures

typedef struct
{
    uint2 base, count, offset;  // (base, tail)
} GroupData;


typedef struct
{
    union
    {
        float3 start;
        float4 start_min;
        struct
        {
            float res1_[3], min;
        };
    };
    union
    {
        float3 dir;
        float4 dir_max;
        struct
        {
            float res2_[3], max;
        };
    };
} Ray;

#define MAX_HITS  64

typedef union
{
    struct
    {
        float pos;
        uint group_id;
        uint2 local_id;
    };
    float4 res_;
} RayHit;

enum RayType
{
    rt_primary, rt_shadow
};

#define MAX_QUEUE_LEN  16

typedef struct
{
    float4 weight;
    uint pixel, type, material_id, queue_len;
    Ray ray;  float3 norm;  RayHit root, orig;
    RayHit queue[MAX_QUEUE_LEN];
} RayQueue;


#define RADIX_SHIFT             5
#define RADIX_MAX       (1 << RADIX_SHIFT)
#define RADIX_MASK        (RADIX_MAX - 1)


#ifdef __cplusplus
#undef uint
#undef uint2
#undef float
#undef float3
#undef float4
#endif
