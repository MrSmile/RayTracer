// ray-tracer.h : OpenCL data layout
//


// material shader

typedef struct
{
    float3 color;
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

typedef struct
{
    uint aabb_offs, count;
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

#define GROUP_ID_MASK  0x00FFFF
#define GROUP_TR_SHIFT       24
#define GROUP_TR_MASK       0xF
#define GROUP_SH_SHIFT       28
#define GROUP_SH_MASK       0xF

enum
{
    tr_none = 0, tr_identity, tr_ortho, tr_affine
};

enum
{
    sh_spawn = 0, sh_sky, sh_material, sh_aabb, sh_mesh
};

enum
{
    spawn_group = 0, sky_group = 1 | sh_sky << GROUP_SH_SHIFT
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
    uint group_count, ray_count, old_count;  // counts must be multiple of UNIT_WIDTH
    Camera cam;
} GlobalData;


// internal data structures

typedef union
{
    struct
    {
        uint res_, cur_index;
    };
    struct
    {
        uint2 count, offset;  // (base, tail)
    };
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

#define MAX_HITS  16

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

typedef struct
{
    RayHit orig;
    float3 norm;
    uint material_id;
} RayStop;

#define QUEUE_ORDER  3
#define MAX_QUEUE_LEN  ((1 << QUEUE_ORDER) - 1)

typedef struct
{
    float4 weight;
    uint pixel, index, queue_len;
    Ray ray;  RayStop stop;  RayHit root;
    RayHit queue[MAX_QUEUE_LEN];
} RayQueue;


#define RADIX_SHIFT             6
#define RADIX_MAX       (1 << RADIX_SHIFT)
#define RADIX_MASK        (RADIX_MAX - 1)
#define SORT_BLOCK             16
#define SORT_WIDTH  (SORT_BLOCK * UNIT_WIDTH)
