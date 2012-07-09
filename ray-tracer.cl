// ray-tracer.cl -- main kernel
//

#define KERNEL  kernel __attribute__((reqd_work_group_size(UNIT_WIDTH, 1, 1)))

#include "ray-tracer.h"
#include "sort.cl"


uint reset_ray(global RayQueue *ray, uint group_id, uint2 local_id)
{
    ray->stop.orig.group_id = ray->queue[0].group_id = group_id;
    ray->stop.orig.local_id = ray->queue[0].local_id = local_id;
    ray->queue[0].pos = ray->ray.min = 0;  ray->ray.max = INFINITY;
    ray->queue_len = 1;  ray->stop.material_id = sky_group;  return group_id;
}


#include "shader.cl"


KERNEL void init_groups(global GroupData *grp_data)
{
    grp_data[get_global_id(0)].cur_index = 0;
}

uint init_ray(const global Camera *cam, global RayQueue *ray, uint pixel)
{
    pixel %= cam->width * cam->height;  // TODO: shuffle
    float x = pixel % cam->width + 0.5, y = pixel / cam->width + 0.5;  // TODO: randomize
    ray->pixel = pixel;  ray->weight = 1;

    ray->ray.start = cam->eye;
    ray->ray.dir = normalize(cam->top_left + x * cam->dx + y * cam->dy);
    return reset_ray(ray, ray->root.group_id = cam->root_group,
        ray->root.local_id = (uint2)(cam->root_local, 0));
}

KERNEL void init_rays(global GlobalData *data, global RayQueue *ray_list, global uint2 *ray_index)
{
    const uint index = get_global_id(0);
    uint group_id = init_ray(&data->cam, &ray_list[index], index);
    ray_index[index] = (uint2)(group_id, index);  if(index)return;
    data->pixel_offset = get_global_size(0);  data->pixel_count = 0;
}

KERNEL void init_image(global float4 *area)
{
    area[get_global_id(0)] = 0;
}


void sort_hits(RayHit *hit, uint n)
{
    hit[n].pos = INFINITY;  // TODO
}

KERNEL void process(global GlobalData *data, global float4 *area,
    global RayQueue *ray_list, global uint2 *ray_index,
    const global Group *grp_list, const global Matrix *mat_list,
    const global AABB *aabb, const global Vertex *vtx, const global uint *tri)
{
    const uint index = get_global_id(0);  if(index >= data->ray_count)return;
    uint offs = ray_index[index].s1;

    global RayQueue *ray = &ray_list[offs];
    uint group_id = ray->queue[0].group_id;

    Ray cur;  float3 mat[4];  uint queue_len, n;
    transform(group_id, ray, &cur, mat, mat_list);
    RayHit hit[MAX_QUEUE_LEN + MAX_HITS + 1];  RayStop stop;
    switch((group_id >> GROUP_SH_SHIFT) & GROUP_SH_MASK)
    {
    case sh_spawn:
        group_id = init_ray(&data->cam, ray, index + data->pixel_offset);  goto assign_index;

    case sh_sky:
        group_id = sky_shader(data, area, ray);  goto assign_index;

    case sh_aabb:
        n = aabb_shader(&cur, &grp_list[group_id & GROUP_ID_MASK].aabb, hit + MAX_QUEUE_LEN, aabb);
        goto insert_hits;

    case sh_mesh:
        if(mesh_shader(&cur, &grp_list[group_id & GROUP_ID_MASK].mesh, ray->queue, &stop, vtx, tri))goto insert_stop;
        break;

    case sh_material:
        group_id = mat_shader(data, area, ray);  goto assign_index;
    }

    queue_len = ray->queue_len - 1;
    for(uint i = 0; i < queue_len; i++)hit[i] = ray->queue[i + 1];

save_queue:
    if(queue_len)
    {
        ray->queue_len = queue_len;  ray->ray.min = ray->queue[0].pos;
    }
    else
    {
        hit[0].group_id = ray->stop.material_id;  hit[0].local_id = 0;  queue_len = 1;
    }

copy_queue:
    for(uint i = 0; i < queue_len; i++)ray->queue[i] = hit[i];  group_id = hit[0].group_id;

assign_index:
    ray_index[index] = (uint2)(group_id & GROUP_ID_MASK, offs);  return;

insert_stop:
    queue_len = ray->queue_len - 1;
    for(uint i = 0; i < queue_len; i++)
    {
        hit[i].pos = ray->queue[i + 1].pos;
        if(hit[i].pos >= stop.orig.pos)
        {
            queue_len = i;  break;
        }
        hit[i].group_id = ray->queue[i + 1].group_id;
        hit[i].local_id = ray->queue[i + 1].local_id;
    }
    if(queue_len)
    {
        ray->queue_len = queue_len;  ray->ray.min = ray->queue[0].pos;
    }
    else
    {
        hit[0].group_id = stop.material_id;  hit[0].local_id = 0;  queue_len = 1;
    }
    stop.norm = mat[0] * stop.norm.x + mat[1] * stop.norm.y + mat[2] * stop.norm.z;
    ray->ray.max = stop.orig.pos;  ray->stop = stop;  goto copy_queue;

insert_hits:
    sort_hits(hit + MAX_QUEUE_LEN, n);
    uint old_len = ray->queue_len;  queue_len = 0;
    for(uint i = 1, j = MAX_QUEUE_LEN; i < old_len; i++)
    {
        float pos = ray->queue[i].pos;
        while(hit[j].pos < pos)
        {
            hit[queue_len++] = hit[j++];
            if(queue_len == MAX_QUEUE_LEN)goto overflow;
        }
        hit[queue_len].pos = pos;
        if(++queue_len == MAX_QUEUE_LEN)goto overflow;
        hit[queue_len - 1].group_id = ray->queue[i].group_id;
        hit[queue_len - 1].local_id = ray->queue[i].local_id;
    }
    goto save_queue;

overflow:
    hit[MAX_QUEUE_LEN - 1].group_id = ray->root.group_id;
    hit[MAX_QUEUE_LEN - 1].local_id = ray->root.local_id;
    goto save_queue;
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
