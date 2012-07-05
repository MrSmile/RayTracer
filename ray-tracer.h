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
    uint vtx_offs, tri_offs, tri_count, material_id;
} MeshShader;


// group description

#define GROUP_ID_MASK  0xFFFFFF
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
    uint group_count, ray_count;  // counts must be multiple of UNIT_WIDTH
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

#define MAX_QUEUE_LEN  9

typedef struct
{
    float4 weight;
    uint pixel, index, queue_len;
    Ray ray;  RayStop stop;  RayHit root;
    RayHit queue[MAX_QUEUE_LEN];
} RayQueue;

typedef float float_unit[UNIT_WIDTH];
typedef uint uint_unit[UNIT_WIDTH];

typedef struct
{
    float_unit pos;
    uint_unit group_id, local0_id, local1_id;
} RayHitUnit;

typedef struct  __attribute__((aligned(128)))
{
    uint_unit index, pixel;
    float_unit weight_r, weight_g, weight_b, weight_w;
    float_unit start_x, start_y, start_z, min;
    float_unit dir_x, dir_y, dir_z, max;
    uint_unit root_group, root_local0, root_local1;
    uint_unit orig_group, orig_local0, orig_local1;
    float_unit norm_x, norm_y, norm_z;
    uint_unit queue_len, material_id;
    RayHitUnit queue[MAX_QUEUE_LEN];
    uint_unit align_[3];
} RayUnit;

#define RAY_UNIT_HEIGHT  (sizeof(RayUnit) / sizeof(float_unit))

typedef union
{
    RayUnit ray;
    uint_unit data[RAY_UNIT_HEIGHT];
} RayUnitData;
