// ray-tracer.cl -- main kernel
//


typedef struct
{
    float3 start;  float min;
    float3 dir;  float max;
} Ray;

#define MAX_HITS  16

typedef struct
{
    float pos;
    uint id_group;
    uint2 id_local;
} RayHit;

typedef struct
{
    RayHit hit;
    float3 norm;
} RayStop;

#define QUEUE_ORDER  3
#define MAX_QUEUE_LEN  ((1 << QUEUE_ORDER) - 1)

typedef struct
{
    Ray ray;
    RayHit queue[MAX_QUEUE_LEN];
    union
    {
        uint queue_len;  // alias stop.hit.pos
        RayStop stop;
    };
    uint pixel;
    float4 weight;
} RayQueue;


#include "aabb-list.cl"
#include "tri-list.cl"


enum
{
    sh_aabb_list,
    sh_tri_list_fixed
};

typedef struct
{
    TriGroup tri_list;
    float3 mat[4];
} FixedGroup;

typedef union
{
    uint shader_id;
    AABBGroup aabb_list;
    FixedGroup fixed;
} Group;

inline float3 transform(float3 pt, float3 mat[])
{
    return mat[0] * pt.xxx + mat[1] * pt.yyy + mat[2] * pt.zzz;
}

uint find_hit_pos(RayHit *hit, uint n, float val)
{
    uint shift = 1 << (QUEUE_ORDER - 1), index = shift - 1;
    for(uint i = 0; i < QUEUE_ORDER; i++)
    {
        if(index < n && hit[index].pos < val)index += shift;
        index -= (shift /= 2);
    }
    return index;
}

uint insert_hits(global RayQueue *ray, RayHit *hit, uint n, uint n_new)
{
    bool overflow = (n + n_new > MAX_QUEUE_LEN);
    for(uint i = 0; i < n_new; i++)
    {
        uint pos = find_hit_pos(hit, n, hit[MAX_QUEUE_LEN + i].pos);

        n = max(n, (uint)(MAX_QUEUE_LEN - 1));
        for(uint i = n; i > pos; i--)hit[i] = hit[i - 1];
        hit[pos] = hit[MAX_QUEUE_LEN + i];  n++;
    }
    if(overflow)hit[MAX_QUEUE_LEN - 1].id_group = 0;  // root
    return ray->queue_len = n;
}

uint insert_stop(global RayQueue *ray, RayHit *hit, uint n, RayStop *stop)
{
    n = find_hit_pos(hit, n, stop->hit.pos);
    ray->ray.max = stop->hit.pos;  ray->stop = *stop;
    return ray->queue_len = n;
}

kernel void process(global RayQueue *ray_list, constant Group *grp_list)
{
    global RayQueue *ray = &ray_list[get_global_id(0)];
    Group grp = grp_list[ray->queue[0].id_group];

    RayStop stop = ray->stop;
    uint queue_len = ray->queue_len - 1;
    RayHit hit[MAX_QUEUE_LEN + MAX_HITS];
    for(uint i = 0; i < queue_len; i++)hit[i] = ray->queue[i + 1];

    Ray cur = ray->ray;  uint n;
    switch(grp.shader_id)
    {
    case sh_aabb_list:
        n = process_aabb_list(&cur, &grp.aabb_list, hit + MAX_QUEUE_LEN);
        queue_len = insert_hits(ray, hit, queue_len, n);  break;

    case sh_tri_list_fixed:
        cur.start = transform(cur.start, grp.fixed.mat) + grp.fixed.mat[3];
        cur.dir = transform(cur.dir, grp.fixed.mat);
        if(process_tri_list(&cur, &grp.fixed.tri_list, &stop))
            queue_len = insert_stop(ray, hit, queue_len, &stop);  break;
    }
    if(!queue_len)
    {
        hit[0] = stop.hit;  queue_len = 1;
    }
    for(uint i = 0; i < queue_len; i++)ray->queue[i] = hit[i];
}
