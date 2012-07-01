

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



typedef struct
{
    float3 min;  uint id_group;
    float3 max;  uint id_local;
} AABB;

typedef struct
{
    uint shader_id, count;
    constant AABB *aabb;
} AABBGroup;

uint process_aabb_list(Ray *ray, AABBGroup *grp, RayHit *hit)
{
    uint hit_count = 0;
    float3 inv_dir = 1 / ray->dir;
    constant AABB *aabb = grp->aabb;
    for(uint i = 0; i < grp->count; i++)
    {
        float3 pos1 = (aabb[i].min - ray->start) * inv_dir;
        float3 pos2 = (aabb[i].max - ray->start) * inv_dir;
        float3 pos_min = min(pos1, pos2), pos_max = max(pos1, pos2);
        float t_min = min(min(pos_min.x, pos_min.y), pos_min.z);
        float t_max = min(min(pos_max.x, pos_max.y), pos_max.z);
        if(!(t_max > ray->min && t_min < ray->max))continue;

        hit[hit_count].pos = t_min;  hit[hit_count].id_group = aabb[i].id_group;
        hit[hit_count].id_local = (uint2)(aabb[i].id_local, 0);  hit_count++;
    }
    return hit_count;
}


typedef struct
{
    float3 pos, norm;
} Vertex;

typedef struct
{
    uint shader_id, count;
    constant Vertex *vtx;
    constant uint *tri;

    uint reserved, id_group;
    uint2 id_local;
} TriGroup;

bool process_tri_list(Ray *ray, TriGroup *grp, RayStop *stop)
{
    float hit_w;
    uint hit_index = 0xFFFFFFFF;
    constant Vertex *vtx = grp->vtx;
    constant uint *tri = grp->tri;
    for(uint i = 0; i < grp->count; i++)
    {
        uint3 index = (tri[i] >> (uint3)(0, 10, 20)) & 0x3FF;
        float3 r = vtx[index.s0].pos, p = vtx[index.s1].pos - r, q = vtx[index.s2].pos - r;  r -= ray->start;
        float3 n = cross(p, q);  float w = 1 / dot(ray->dir, n), t = dot(r, n) * w;  // w sign -- cull mode
        if(!(t > ray->min && t < ray->max))continue;  stop->hit.pos = t;  hit_w = w;  hit_index = i;
    }
    if(hit_index == 0xFFFFFFFF)return false;

    uint3 index = (tri[hit_index] >> (uint3)(0, 10, 20)) & 0x3FF;
    float3 r = vtx[index.s0].pos, p = vtx[index.s1].pos - r, q = vtx[index.s2].pos - r;  r -= ray->start;
    float3 dr = cross(ray->dir, r);  float u = -dot(q, dr) * hit_w, v = dot(p, dr) * hit_w;
    stop->norm = vtx[index.s0].norm * (1 - u - v) + vtx[index.s1].norm * u + vtx[index.s2].norm * v;
    stop->hit.id_group = grp->id_group;  stop->hit.id_local = grp->id_local;  return true;
}


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
