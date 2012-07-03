// ray-tracer.h : OpenCL data layout
//


// AABB shader

typedef union
{
    float3 min, max;
    struct
    {
        float res1_[3], group_id;
        float res2_[3], local_id;
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
    uint vtx_offs, tri_offs, tri_count;
    uint material_id;
} MeshShader;


// material shader

typedef struct
{
    uint shader_id;
    float3 color;
} MatShader;


// group description

enum
{
    sh_sky, sh_aabb, sh_mesh, sh_material
};

enum
{
    tr_identity, tr_ortho, tr_affine, tr_none
};

typedef struct
{
    float4 x, y, z;
} Matrix;

typedef struct
{
    uint transform_id, shader_id;
    union
    {
        AABBShader aabb;
        MeshShader mesh;
        MatShader material;
    };
} Group;


// global scene data

typedef struct
{
    float3 eye, top_left, dx, dy;
    uint width, height, root_group, root_local;
} Camera;

typedef struct
{
    uint cur_pixel, group_count, ray_count;  // counts must be multiple of UNIT_WIDTH
    Camera cam;
} GlobalData;


// internal data structures

typedef struct
{
    uint cur_index, base_count;
    uint2 offset;  // (base, tail)
} GroupData;


typedef union
{
    float3 start, dir;
    struct
    {
        float res1_[3], min;
        float res2_[3], max;
    };
} Ray;

#define MAX_HITS  16

typedef struct
{
    float pos;
    uint group_id;
    uint2 local_id;
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
} RayHeader;

typedef struct
{
    RayHeader hdr;
    RayHit queue[MAX_QUEUE_LEN];
} RayQueue;
