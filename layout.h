// layout.h : OpenCL data layout
//


typedef union
{
    cl_float3 min, max;
    struct
    {
        cl_float res1_[3], group_id;
        cl_float res2_[3], local_id;
    };
} AABB;

typedef struct
{
    cl_uint aabb_offs, count;
} AABBShader;


typedef struct
{
    cl_float3 pos, norm;
} Vertex;

typedef struct
{
    cl_uint vtx_offs, tri_offs, tri_count;
    cl_uint material_id;
} TriShader;


typedef struct
{
    cl_uint shader_id;
    cl_float3 color;
} MatShader;


enum
{
    sh_aabb_list, sh_tri_list, sh_material
};

enum
{
    tr_identity, tr_ortho, tr_affine, tr_none
};

typedef struct
{
    cl_float4 x, y, z;
} Matrix;

typedef struct
{
    cl_uint transform_id, shader_id;
    union
    {
        AABBShader aabb_list;
        TriShader tri_list;
        MatShader material;
    };
} Group;
