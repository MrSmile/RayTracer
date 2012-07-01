// layout.h : OpenCL data layout
//


typedef union
{
    cl_float3 min, max;
    struct
    {
        cl_float res1_[3], id_group;
        cl_float res2_[3], id_local;
    };
} AABB;

typedef struct
{
    cl_uint shader_id, aabb_offs, count;
} AABBGroup;


typedef struct
{
    cl_float3 pos, norm;
} Vertex;

typedef struct
{
    cl_uint shader_id, vtx_offs, tri_offs, tri_count;
    cl_uint id_group;  cl_uint2 id_local;
} TriGroup;


enum
{
    sh_aabb_list,
    sh_tri_list_fixed
};

typedef struct
{
    TriGroup tri_list;
    cl_float3 mat[4];
} FixedGroup;

typedef union
{
    cl_uint shader_id;
    AABBGroup aabb_list;
    FixedGroup fixed;
} Group;
