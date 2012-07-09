// ray-tracer.cl -- main kernel
//

#define KERNEL  kernel __attribute__((reqd_work_group_size(UNIT_WIDTH, 1, 1)))

#include "ray-tracer.h"
#include "sort.cl"


void reset_ray(RayHeader *ray, RayHit *hit)
{
    ray->ray.min = 0;  ray->ray.max = INFINITY;
    ray->stop.orig = hit[0] = ray->root;
    ray->stop.material_id = sky_group;
    ray->queue_len = 1;
}


#include "shader.cl"


KERNEL void init_groups(global GroupData *grp_data)
{
    grp_data[get_global_id(0)].cur_index = 0;
}

void init_ray(const Camera *cam, RayHeader *ray, RayHit *hit, uint pixel)
{
    pixel %= cam->width * cam->height;  // TODO: shuffle
    float x = pixel % cam->width + 0.5, y = pixel / cam->width + 0.5;  // TODO: randomize
    ray->pixel = pixel;  ray->weight = 1;

    ray->ray.start = cam->eye;
    ray->ray.dir = normalize(cam->top_left + x * cam->dx + y * cam->dy);
    ray->root.pos = 0;  ray->root.group_id = cam->root_group;
    ray->root.local_id = (uint2)(cam->root_local, 0);
    reset_ray(ray, hit);
}

KERNEL void init_rays(global GlobalData *data, global RayQueue *ray_list, global uint2 *ray_index)
{
    Camera cam = data->cam;  RayHeader ray;  RayHit hit;
    const uint index = get_global_id(0);  init_ray(&cam, &ray, &hit, index);
    ray_list[index].hdr = ray;  ray_list[index].queue[0] = hit;
    ray_index[index] = (uint2)(hit.group_id, index);  if(index)return;
    data->pixel_offset = get_global_size(0);  data->pixel_count = 0;
}

KERNEL void init_image(global float4 *area)
{
    area[get_global_id(0)] = 0;
}


uint find_hit_pos(const RayHit *hit, uint n, float val)
{
    uint shift = 1 << (QUEUE_ORDER - 1), index = shift - 1;
    for(uint i = 0; i < QUEUE_ORDER; i++)
    {
        if(index < n && hit[index].pos < val)index += shift;
        index -= (shift /= 2);
    }
    return index;
}

void insert_hits(RayHeader *ray, RayHit *hit, uint n)
{
    bool overflow = (ray->queue_len + n > MAX_QUEUE_LEN);
    for(uint i = 0; i < n; i++)
    {
        uint pos = find_hit_pos(hit, ray->queue_len, hit[MAX_QUEUE_LEN + i].pos);
        for(uint i = ray->queue_len; i > pos; i--)hit[i] = hit[i - 1];
        hit[pos] = hit[MAX_QUEUE_LEN + i];

        ray->queue_len = min(ray->queue_len + 1, (uint)MAX_QUEUE_LEN - 1);
    }
    if(overflow)
    {
        hit[MAX_QUEUE_LEN - 1].group_id = ray->root.group_id;
        hit[MAX_QUEUE_LEN - 1].local_id = ray->root.local_id;
    }
}

void insert_stop(RayHeader *ray, RayHit *hit, const float3 mat[4])
{
    ray->queue_len = find_hit_pos(hit, ray->queue_len, ray->stop.orig.pos);
    ray->stop.norm = mat[0] * ray->stop.norm.x + mat[1] * ray->stop.norm.y + mat[2] * ray->stop.norm.z;
    ray->ray.max = ray->stop.orig.pos;
}

uint debug_test(const global Group *grp_list)  // DEBUG
{
    Group grp = *grp_list;  return grp.mesh.material_id;
}

KERNEL void process(global GlobalData *data, global float4 *area,
    global RayQueue *ray_list, global uint2 *ray_index,
    const global Group *grp_list, const global Matrix *mat_list,
    const global AABB *aabb, const global Vertex *vtx, const global uint *tri)
{
    const uint index = get_global_id(0);  if(index >= data->ray_count)return;
    uint offs = ray_index[index].s1;

    global RayQueue *ptr = &ray_list[offs];
    RayHeader ray = ptr->hdr;  ray.queue_len--;
    RayHit hit[MAX_QUEUE_LEN + MAX_HITS], cur_hit = ptr->queue[0];
    for(uint i = 0; i < ray.queue_len; i++)hit[i] = ptr->queue[i + 1];
    const global Group *grp = &grp_list[cur_hit.group_id & GROUP_ID_MASK];
    ray.ray.min = cur_hit.pos;

    Ray cur = ray.ray;  float3 mat[4];  uint n;
    transform(&cur, &cur_hit, cur_hit.group_id, mat_list, mat);
    switch((cur_hit.group_id >> GROUP_SH_SHIFT) & GROUP_SH_MASK)
    {
    case sh_spawn:
        {
            Camera cam = data->cam;
            init_ray(&cam, &ray, &hit, index + data->pixel_offset);  break;
        }

    case sh_sky:
        sky_shader(data, area, &ray, hit);  break;

    case sh_material:
        mat_shader(data, area, &ray, hit);  break;

    case sh_aabb:
        n = aabb_shader(&cur, &grp->aabb, hit + MAX_QUEUE_LEN, aabb);
        insert_hits(&ray, hit, n);  break;

    case sh_mesh:
        if(mesh_shader(&cur, &grp->mesh, cur_hit, &ray.stop, vtx, tri))
            insert_stop(&ray, hit, mat);  break;
    }
    if(!ray.queue_len)
    {
        hit[0].pos = ray.stop.orig.pos;
        hit[0].group_id = ray.stop.material_id;
        hit[0].local_id = 0;  ray.queue_len = 1;
    }
    ray_index[index] = (uint2)(hit[0].group_id & GROUP_ID_MASK, offs);  ptr->hdr = ray;
    for(uint i = 0; i < ray.queue_len; i++)ptr->queue[i] = hit[i];
}


KERNEL void count_groups(global GlobalData *data,
    global GroupData *grp_data, const global uint2 *ray_index)
{
    const uint index = get_local_id(0), offs = get_global_id(0);
    local buf[UNIT_WIDTH];  buf[index] = ray_index[offs].s0;  barrier(CLK_LOCAL_MEM_FENCE);
    uint prev, next, pos;
    if(offs)
    {
        prev = index ? buf[index - 1] : ray_index[offs - 1].s0;
        next = buf[index];  pos = offs;
    }
    else
    {
        const uint total = get_global_size(0);
        prev = ray_index[total - 1].s0;
        next = data->group_count;  pos = total;
    }
    for(; prev < next; prev++)grp_data[prev].cur_index = pos;
}

KERNEL void update_groups(global GlobalData *data, global GroupData *grp_data)  // single unit
{
    const uint ray_count = data->ray_count;  uint2 offs = 0;
    const uint index = get_global_id(0), cur = UNIT_WIDTH + index, n = data->group_count;
    local uint2 buf[2 * UNIT_WIDTH];  buf[index] = 0;  uint prev = 0;
    for(uint pos = index; pos < n; pos += UNIT_WIDTH)
    {
        GroupData grp;
        grp.count.s0 = buf[cur].s0 = grp_data[pos].cur_index;
        barrier(CLK_LOCAL_MEM_FENCE);  uint base;
        if(index)base = buf[cur - 1].s0;
        else
        {
            base = prev;  prev = buf[2 * UNIT_WIDTH - 1].s0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);  grp.count.s0 -= base;

        grp.count.s1 = grp.count.s0 % UNIT_WIDTH;
        grp.count.s0 -= grp.count.s1;  uint2 res = grp.count;
        for(uint offs = 1; offs < UNIT_WIDTH; offs *= 2)
        {
            buf[cur] = res;  barrier(CLK_LOCAL_MEM_FENCE);
            res += buf[cur - offs];  barrier(CLK_LOCAL_MEM_FENCE);
        }
        buf[cur] = res;  barrier(CLK_LOCAL_MEM_FENCE);

        grp.offset = offs + res - grp.count;  offs += buf[2 * UNIT_WIDTH - 1];
        grp.count.s1 = base;  grp_data[pos] = grp;  if(pos)continue;

        data->pixel_offset += data->pixel_count;  data->pixel_count = grp.count.s0;
    }
    for(uint pos = index; pos < n; pos += UNIT_WIDTH)grp_data[pos].offset.s1 += offs.s0;
    if(index)return;  data->ray_count = offs.s0;  data->old_count = ray_count;
}

KERNEL void set_ray_index(const global GroupData *grp_data,
    const global uint2 *src_index, global uint2 *dst_index)
{
    const uint index = get_global_id(0);  uint2 src = src_index[index];
    GroupData grp = grp_data[src.s0];  uint offs = index - grp.count.s1;
    if(offs < grp.count.s0)offs += grp.offset.s0;
    else offs += grp.offset.s1 - grp.count.s0;
    dst_index[offs] = src;
}
