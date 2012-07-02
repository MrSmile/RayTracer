// ray-tracer.cl -- main kernel
//

#include "ray-tracer.h"
#include "shader.cl"


kernel void init(global GlobalData *data, global GroupData *grp_data, global RayQueue *ray_list)
{
    const uint index = get_global_id(0), size = get_global_size(0), n = data->group_count;
    for(uint pos = index; pos < n; pos += size)grp_data[pos].cur_index = 0;

    Camera cam = data->cam;  RayQueue ray;
    init_ray(&ray, &cam, index);  ray_list[index] = ray;
}


uint find_hit_pos(const RayHit *hit, uint n, float val)
{
    uint shift = 1 << (QUEUE_ORDER - 1), index = shift;
    for(uint i = 0; i < QUEUE_ORDER; i++)
    {
        if(index < n && hit[index].pos < val)index += shift;
        index -= (shift /= 2);
    }
    return index;
}

void insert_hits(RayQueue *ray, uint n)
{
    bool overflow = (ray->queue_len + n > QUEUE_OFFSET);
    for(uint i = 0; i < n; i++)
    {
        uint pos = find_hit_pos(ray->queue, ray->queue_len, ray->queue[QUEUE_OFFSET + i].pos);
        for(uint i = ray->queue_len; i > pos; i--)ray->queue[i] = ray->queue[i - 1];
        ray->queue[pos] = ray->queue[QUEUE_OFFSET + i];

        ray->queue_len = min(ray->queue_len + 1, (uint)MAX_QUEUE_LEN);
    }
    if(overflow)
    {
        ray->queue[MAX_QUEUE_LEN].group_id = ray->root.group_id;
        ray->queue[MAX_QUEUE_LEN].local_id = ray->root.local_id;
    }
}

void insert_stop(RayQueue *ray, const float3 mat[4])
{
    ray->queue_len = find_hit_pos(ray->queue, ray->queue_len, ray->stop.orig.pos) - 1;
    ray->stop.norm = mat[0] * ray->stop.norm.x + mat[1] * ray->stop.norm.y + mat[2] * ray->stop.norm.z;
    ray->ray.max = ray->stop.orig.pos;
}

kernel void process(global GlobalData *data, global GroupData *grp_data, global RayQueue *ray_list,
    global const Group *grp_list, global const Matrix *mat_list,
    global const AABB *aabb, global const Vertex *vtx, global const uint *tri)
{
    union
    {
        RayQueue ray;
        char padding_[sizeof(RayQueue) + (MAX_HITS + 1) * sizeof(RayHit)];
    } ray_data = {ray_list[get_global_id(0)]};

    RayQueue *ray = &ray_data.ray;
    Group grp = grp_list[ray->queue[0].group_id];

    Ray cur = ray->ray;  float3 mat[3];  uint n;
    transform(&cur, ray->queue, &grp, mat_list, mat);
    switch(grp.shader_id)
    {
    case sh_sky:
        sky_shader(ray, data);  break;

    case sh_aabb:
        n = aabb_shader(&cur, &grp.aabb, ray->queue + QUEUE_OFFSET, aabb);
        insert_hits(ray, n);  break;

    case sh_mesh:
        if(mesh_shader(&cur, &grp.mesh, ray->queue[0], &ray->stop, vtx, tri))
            insert_stop(ray, mat);  break;

    case sh_material:
        mat_shader(ray, data);  break;
    }
    ray->ray.min = ray->queue[0].pos;
    if(!--ray->queue_len)
    {
        ray->queue[1].pos = ray->stop.orig.pos;
        ray->queue[1].group_id = ray->stop.material_id;
        ray->queue[1].local_id = 0;  ray->queue_len = 1;
    }

    ray->index = atomic_add(&grp_data[ray->queue[1].group_id].cur_index, 1);  // TODO: optimize

    ray_list[get_global_id(0)] = *ray;
}


kernel void update_groups(global GlobalData *data, global GroupData *grp_data)
{
    const uint index = get_global_id(0), size = get_global_size(0), n = data->group_count;

    uint base = 0, tail = 0;
    for(uint pos = index; pos < n; pos += size)
    {
        uint n = grp_data[pos].cur_index;  GroupData data;
        data.cur_index = 0;  data.tail_offs = n % size;
        data.base_offs = data.base_count = n - data.tail_offs;

        // TODO: sum

        grp_data[pos] = data;
    }
    for(uint pos = index; pos < n; pos += size)
        grp_data[pos].tail_offs += base - grp_data[pos].base_count;
    if(!index)data->ray_count = base;
}

kernel void shuffle_rays(global GroupData *grp_data, const global RayQueue *src, global RayQueue *dst)
{
    RayQueue ray = src[get_global_id(0)];
    GroupData data = grp_data[ray.queue[1].group_id];
    uint offs = ray.index < data.base_count ? data.base_offs : data.tail_offs;
    dst[offs + ray.index] = ray;
}
