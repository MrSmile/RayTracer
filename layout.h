// layout.h : OpenCL data layout
//

enum
{
    tr_identity, tr_matrix, tr_none
};

typedef cl_float4 Matrix[3];

typedef union
{
    Matrix mat;
} Transform;


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

typedef union
{
    AABBShader aabb_list;
    TriShader tri_list;
    MatShader material;
} Shader;

typedef struct
{
    cl_uint transform_id, shader_id;
    Transform trans;
    Shader shader;
} Group;
