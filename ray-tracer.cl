// ray-tracer.cl -- main kernel
//


typedef uint cl_uint;
typedef uint2 cl_uint2;
typedef float cl_float;
typedef float3 cl_float3;
typedef float4 cl_float4;
#include "layout.h"


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


#include "shader.cl"


#define QUEUE_ORDER  3
#define QUEUE_OFFSET  (1 << QUEUE_ORDER)
#define MAX_QUEUE_LEN  (QUEUE_OFFSET - 1)

typedef struct
{
    Ray ray;
    RayHit root, queue[MAX_QUEUE_LEN];
    union
    {
        uint queue_len;  // alias stop.orig.pos
        RayStop stop;
    };
    uint pixel;
    float4 weight;
} RayQueue;


uint find_hit_pos(RayHit *hit, uint n, float val)
{
    uint shift = 1 << (QUEUE_ORDER - 1), index = shift;
    for(uint i = 0; i < QUEUE_ORDER; i++)
    {
        if(index < n && hit[index].pos < val)index += shift;
        index -= (shift /= 2);
    }
    return index;
}

uint insert_hits(global RayHit *root, RayHit *hit, uint n, uint n_new)
{
    bool overflow = (n + n_new > QUEUE_OFFSET);
    for(uint i = 0; i < n_new; i++)
    {
        uint pos = find_hit_pos(hit, n, hit[QUEUE_OFFSET + i].pos);

        n = max(n, (uint)MAX_QUEUE_LEN);
        for(uint i = n; i > pos; i--)hit[i] = hit[i - 1];
        hit[pos] = hit[QUEUE_OFFSET + i];  n++;
    }
    if(overflow)
    {
        hit[MAX_QUEUE_LEN].group_id = root->group_id;
        hit[MAX_QUEUE_LEN].local_id = root->local_id;
    }
    return n - 1;
}

uint insert_stop(global RayQueue *ray, RayHit *hit, uint n, RayStop *stop, float3 mat[4])
{
    n = find_hit_pos(hit, n, stop->orig.pos);
    stop->norm = mat[0] * stop->norm.x + mat[1] * stop->norm.y + mat[2] * stop->norm.z;
    ray->ray.max = stop->orig.pos;  ray->stop = *stop;  return n - 1;
}

kernel void process(global RayQueue *ray_list, global const Group *grp_list, global const Matrix *mat_list,
    global const AABB *aabb, global const Vertex *vtx, global const uint *tri)
{
    global RayQueue *ray = &ray_list[get_global_id(0)];

    uint queue_len = ray->queue_len;
    RayHit hit[QUEUE_OFFSET + MAX_HITS];
    for(uint i = 0; i < queue_len; i++)hit[i] = ray->queue[i];
    Group grp = grp_list[hit[0].group_id];
    RayStop stop = ray->stop;

    Ray cur = ray->ray;  float3 mat[3];  uint n;
    transform(&cur, hit, &grp, mat_list, mat);
    switch(grp.shader_id)
    {
    case sh_aabb_list:
        n = process_aabb_list(&cur, &grp.aabb_list, hit + QUEUE_OFFSET, aabb);
        queue_len = insert_hits(&ray->root, hit, queue_len, n);  break;

    case sh_tri_list:
        if(process_tri_list(&cur, &grp.tri_list, hit[0], &stop, vtx, tri))
            queue_len = insert_stop(ray, hit, queue_len, &stop, mat);  break;
    }
    ray->ray.min = hit[0].pos;
    if(!queue_len)
    {
        hit[1].pos = stop.orig.pos;
        hit[1].group_id = stop.material_id;
        hit[1].local_id = (uint2)(0, 0);
        queue_len = 1;
    }
    for(uint i = 0; i < queue_len; i++)ray->queue[i] = hit[i + 1];
}
