// ray-tracer.cl -- main kernel
//

#include "ray-tracer.h"
#include "shader.cl"


kernel __attribute__((reqd_work_group_size(UNIT_WIDTH, 1, 1)))
    void init(global GlobalData *data, global GroupData *grp_data, global RayQueue *ray_list)
{
    const uint index = get_global_id(0), n = data->group_count;
    for(uint pos = index; pos < n; pos += UNIT_WIDTH)grp_data[pos].cur_index = 0;

    Camera cam = data->cam;  RayHeader ray;  RayHit hit;
    init_ray(&ray, &hit, &cam, index);
    ray_list[index].hdr = ray;  ray_list[index].queue[0] = hit;
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

void insert_hits(RayHeader *ray, RayHit *hit, uint n)
{
    bool overflow = (ray->queue_len + n > QUEUE_OFFSET);
    for(uint i = 0; i < n; i++)
    {
        uint pos = find_hit_pos(hit, ray->queue_len, hit[QUEUE_OFFSET + i].pos);
        for(uint i = ray->queue_len; i > pos; i--)hit[i] = hit[i - 1];
        hit[pos] = hit[QUEUE_OFFSET + i];

        ray->queue_len = min(ray->queue_len + 1, (uint)MAX_QUEUE_LEN);
    }
    if(overflow)
    {
        hit[MAX_QUEUE_LEN].group_id = ray->root.group_id;
        hit[MAX_QUEUE_LEN].local_id = ray->root.local_id;
    }
}

void insert_stop(RayHeader *ray, RayHit *hit, const float3 mat[4])
{
    ray->queue_len = find_hit_pos(hit, ray->queue_len, ray->stop.orig.pos) - 1;
    ray->stop.norm = mat[0] * ray->stop.norm.x + mat[1] * ray->stop.norm.y + mat[2] * ray->stop.norm.z;
    ray->ray.max = ray->stop.orig.pos;
}

kernel __attribute__((reqd_work_group_size(UNIT_WIDTH, 1, 1)))
    void process(global GlobalData *data, global GroupData *grp_data, global RayQueue *ray_list,
        global const Group *grp_list, global const Matrix *mat_list,
        global const AABB *aabb, global const Vertex *vtx, global const uint *tri)
{
    global RayQueue *ptr = &ray_list[get_global_id(0)];
    RayHeader ray = ptr->hdr;  RayHit hit[QUEUE_OFFSET + MAX_HITS];
    for(uint i = 0; i < ray.queue_len; i++)hit[i] = ptr->queue[i];
    Group grp = grp_list[hit[0].group_id];

    Ray cur = ray.ray;  float3 mat[3];  uint n;
    transform(&cur, hit, &grp, mat_list, mat);
    switch(grp.shader_id)
    {
    case sh_sky:
        sky_shader(&ray, hit, data);  break;

    case sh_aabb:
        n = aabb_shader(&cur, &grp.aabb, hit + QUEUE_OFFSET, aabb);
        insert_hits(&ray, hit, n);  break;

    case sh_mesh:
        if(mesh_shader(&cur, &grp.mesh, hit[0], &ray.stop, vtx, tri))
            insert_stop(&ray, hit, mat);  break;

    case sh_material:
        mat_shader(&ray, hit, data);  break;
    }
    ray.ray.min = hit[0].pos;
    if(!--ray.queue_len)
    {
        hit[1].pos = ray.stop.orig.pos;
        hit[1].group_id = ray.stop.material_id;
        hit[1].local_id = 0;  ray.queue_len = 1;
    }

    ray.index = atomic_add(&grp_data[hit[1].group_id].cur_index, 1);  // TODO: optimize

    ptr->hdr = ray;
    for(uint i = 0; i < ray.queue_len; i++)ptr->queue[i] = hit[i + 1];
}


kernel __attribute__((reqd_work_group_size(UNIT_WIDTH, 1, 1)))
    void update_groups(global GlobalData *data, global GroupData *grp_data)  // single unit
{
    local uint2 buf[2 * UNIT_WIDTH], *ptr = buf + UNIT_WIDTH;
    const uint index = get_global_id(0), n = data->group_count;  uint2 offs = 0;
    for(uint pos = index; pos < n; pos += UNIT_WIDTH)
    {
        uint base = grp_data[pos].cur_index, tail = base % UNIT_WIDTH;
        buf[index] = 0;  uint cur = UNIT_WIDTH + index;
        buf[cur] = (uint2)(base -= tail, tail);

        for(uint offs = 1; offs < UNIT_WIDTH; offs *= 2)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            uint2 res = buf[cur] + buf[cur - offs];
            barrier(CLK_LOCAL_MEM_FENCE);
            buf[cur] = res;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        GroupData data;  data.cur_index = 0;  data.base_count = base;
        data.offset = offs + buf[cur - 1];  offs += buf[2 * UNIT_WIDTH - 1];
        grp_data[pos] = data;
    }
    for(uint pos = index; pos < n; pos += UNIT_WIDTH)
        grp_data[pos].offset.s1 += offs.s0 - grp_data[pos].base_count;
    if(!index)data->ray_count = offs.s0;
}

kernel __attribute__((reqd_work_group_size(UNIT_WIDTH, 1, 1)))
    void shuffle_rays(global GroupData *grp_data, const global RayQueue *src, global RayQueue *dst)
{
    RayQueue ray = src[get_global_id(0)];
    GroupData data = grp_data[ray.queue[1].group_id];
    uint offs = ray.hdr.index < data.base_count ? data.offset.s0 : data.offset.s1;
    dst[offs + ray.hdr.index] = ray;
}
