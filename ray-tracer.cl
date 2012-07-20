// ray-tracer.cl -- main kernel
//

#define KERNEL  kernel __attribute__((reqd_work_group_size(UNIT_WIDTH, 1, 1)))

#include "ray-tracer.h"
#include "sort.cl"


uint reset_ray(global RayQueue *ray, uint group_id, uint2 local_id, uint end_group)
{
    ray->orig.group_id = ray->queue[0].group_id = group_id;
    ray->orig.local_id = ray->queue[0].local_id = local_id;
    ray->queue[0].pos = ray->ray.min = 0.001;  ray->ray.max = INFINITY;
    ray->queue_len = 1;  ray->material_id = end_group;  return group_id;
}


#include "shader.cl"


KERNEL void init_groups(global GroupData *grp_data)
{
    const uint index = get_global_id(0);  grp_data[index].count.s1 = 0;
    grp_data[index].offset.s1 = 0xFFFFFFFF;
}

uint calc_crc(uint val)  // TODO: optimize
{
    const uint poly = 0x04C11DB7;
    for(uint i = 0; i < 32; i++)
    {
        uint val_xor = val & 0x80000000 ? poly : 0;
        val = val << 1 ^ val_xor;
    }
    return val;
}

uint2 deinterleave(uint val)
{
    uint2 res = (uint2)(val, val >> 1) & 0x55555555;
    res = (res | res >> 1) & 0x33333333;
    res = (res | res >> 2) & 0x0F0F0F0F;
    res = (res | res >> 4) & 0x00FF00FF;
    res = (res | res >> 8) & 0x0000FFFF;
    return res;
}

float subpixel(uint val)
{
    val = (val % 15) + 1;
    val = (val & 1) << 3 | (val & 2) << 1 | (val & 4) >> 1 | (val & 8) >> 3;
    return (val - 0.5) / 15;
}

uint init_ray(const global GlobalData *data, global RayQueue *ray, uint pixel)
{
    //pixel = calc_crc(pixel);
    const global Camera *cam = &data->cam;
    const uint total = cam->width * cam->height;
    //if(pixel >= total)return data->group_count - 1;  // dead ray
    uint2 sub = deinterleave(pixel / total);  pixel %= total;
    float x = pixel % cam->width + subpixel(sub.x), y = pixel / cam->width + subpixel(sub.y);

    //pixel %= cam->width * cam->height;
    //pixel = calc_crc(pixel) % (cam->width * cam->height);
    //float x = pixel % cam->width + 0.5, y = pixel / cam->width + 0.5;
    ray->pixel = pixel;  ray->weight = 1;

    ray->type = rt_primary;  ray->ray.start_min.xyz = cam->eye;
    ray->ray.dir_max.xyz = normalize(cam->top_left + x * cam->dx + y * cam->dy);
    return reset_ray(ray, ray->root.group_id = cam->root_group,
        ray->root.local_id = (uint2)(cam->root_local, 0), sky_group);
}

KERNEL void init_rays(global GlobalData *data, global RayQueue *ray_list, global uint2 *ray_index)
{
    const uint index = get_global_id(0);
    uint group_id = init_ray(data, &ray_list[index], index);
    ray_index[index] = (uint2)(group_id, index);  if(index)return;
    data->pixel_offset = get_global_size(0);  data->pixel_count = 0;
}

KERNEL void init_image(global float4 *area)
{
    area[get_global_id(0)] = 0;
}


void bitonic_flip(RayHit *hit, uint offs, uint n)
{
    for(uint i = offs; i < n; i++)if(i & offs)
    {
        uint j = i - 2 * (i & (offs - 1)) - 1;
        if(!(hit[j].pos > hit[i].pos))continue;
        RayHit tmp = hit[j];  hit[j] = hit[i];  hit[i] = tmp;
    }
}

void bitonic_step(RayHit *hit, uint offs, uint n)
{
    for(uint i = offs; i < n; i++)if(i & offs)
    {
        uint j = i - offs;
        if(!(hit[j].pos > hit[i].pos))continue;
        RayHit tmp = hit[j];  hit[j] = hit[i];  hit[i] = tmp;
    }
}

void sort_hits(RayHit *hit, uint n)
{
    for(uint base = 1; base < n; base *= 2)
    {
        bitonic_flip(hit, base, n);
        for(uint offs = base / 2; offs; offs /= 2)bitonic_step(hit, offs, n);
    }
}

KERNEL void process(global GlobalData *data, global float4 *area,
    global RayQueue *ray_list, global uint2 *ray_index,
    const global Group *grp_list, const global Matrix *mat_list,
    const global AABB *aabb, const global Vertex *vtx, const global uint *tri)
{
    const uint index = get_global_id(0);  if(index >= data->ray_count)return;
    uint group_id  = ray_index[index].s0, offs = ray_index[index].s1;
    global RayQueue *ray = &ray_list[offs];

    Ray cur;  float3 mat[4];  uint queue_len, n, material_id;
    transform(group_id, ray, &cur, mat, mat_list);
    RayHit hit[MAX_QUEUE_LEN], new_hit[MAX_HITS];  float4 norm_pos;
    switch((group_id >> GROUP_SH_SHIFT) & GROUP_SH_MASK)
    {
    case sh_spawn:
        group_id = init_ray(data, ray, index + data->pixel_offset);  goto assign_index;

    case sh_sky:
        group_id = sky_shader(area, ray, &grp_list[group_id & GROUP_ID_MASK].material);  goto assign_index;

    case sh_light:
        group_id = light_shader(area, ray, &grp_list[group_id & GROUP_ID_MASK].material);  goto assign_index;

    case sh_material:
        group_id = mat_shader(area, ray, &grp_list[group_id & GROUP_ID_MASK].material);  goto assign_index;

    case sh_aabb:
        n = aabb_shader(&cur, &grp_list[group_id & GROUP_ID_MASK].aabb, ray->queue, new_hit, aabb);
        goto insert_hits;

    case sh_mesh:
        material_id = mesh_shader(&cur, &grp_list[group_id & GROUP_ID_MASK].mesh, &norm_pos, vtx, tri);
        if(material_id != 0xFFFFFFFF)goto insert_stop;  break;
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
        hit[0].group_id = ray->material_id;  hit[0].local_id = 0;  queue_len = 1;
    }

copy_queue:
    for(uint i = 0; i < queue_len; i++)ray->queue[i] = hit[i];  group_id = hit[0].group_id;

assign_index:
    ray_index[index] = (uint2)(group_id, offs);  return;

insert_stop:
    if(ray->type == rt_shadow)
    {
        area[ray->pixel] += (float4)(0, 0, 0, ray->weight.w);  material_id = spawn_group;
    }
    else
    {
        ray->norm = mat[0] * norm_pos.x + mat[1] * norm_pos.y + mat[2] * norm_pos.z;
        RayHit orig = {ray->ray.max = norm_pos.w, ray->queue[0].group_id, ray->queue[0].local_id};
        ray->orig = orig;
    }
    queue_len = ray->queue_len - 1;
    for(uint i = 0; i < queue_len; i++)
    {
        hit[i].pos = ray->queue[i + 1].pos;
        if(hit[i].pos >= norm_pos.w)
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
        hit[0].group_id = material_id;  hit[0].local_id = 0;  queue_len = 1;
    }
    ray->material_id = material_id;  goto copy_queue;

insert_hits:
    sort_hits(new_hit, n);  queue_len = 0;
    uint old_len = ray->queue_len, next = 0;
    for(uint i = 1; i < old_len; i++)
    {
        float pos = ray->queue[i].pos;
        while(next < n && new_hit[next].pos < pos)
        {
            hit[queue_len++] = new_hit[next++];
            if(queue_len == MAX_QUEUE_LEN)goto overflow;
        }
        hit[queue_len].pos = pos;
        if(++queue_len == MAX_QUEUE_LEN)goto overflow;
        hit[queue_len - 1].group_id = ray->queue[i].group_id;
        hit[queue_len - 1].local_id = ray->queue[i].local_id;
    }
    while(next < n)
    {
        hit[queue_len++] = new_hit[next++];
        if(queue_len == MAX_QUEUE_LEN)goto overflow;
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
    const uint total = data->ray_count;  if(offs >= total)return;
    local uint buf[UNIT_WIDTH];  buf[index] = ray_index[offs].s0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint prev, next;
    if(index)prev = buf[index - 1];
    else if(offs)prev = ray_index[offs - 1].s0;
    else
    {
        prev = ray_index[total - 1].s0 & GROUP_ID_MASK;  next = data->group_count;
        for(; prev < next; prev++)grp_data[prev].count.s0 = total;  prev = 0;
    }
    prev &= GROUP_ID_MASK;  next = buf[index] & GROUP_ID_MASK;
    for(; prev < next; prev++)grp_data[prev].count.s0 = offs;
}

KERNEL void update_groups(global GlobalData *data, global GroupData *grp_data)  // single unit
{
    const uint index = get_global_id(0), cur = UNIT_WIDTH + index, n = data->group_count;
    local uint2 buf[2 * UNIT_WIDTH];  buf[index] = 0;  uint2 offset = 0;  uint prev = 0;
    for(uint pos = index; pos < n; pos += UNIT_WIDTH)
    {
        GroupData grp;
        grp.count.s0 = buf[cur].s0 = grp_data[pos].count.s0;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(index)grp.base.s0 = buf[cur - 1].s0;
        else
        {
            grp.base.s0 = prev;  prev = buf[2 * UNIT_WIDTH - 1].s0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);  grp.count.s0 -= grp.base.s0;
        grp.base.s1 = grp_data[pos].offset.s1 - grp.count.s0;
        grp.count.s0 += grp_data[pos].count.s1;

        grp.count.s1 = grp.count.s0;
        if(pos != n - 1)grp.count.s1 %= WARP_WIDTH;
        grp.count.s0 -= grp.count.s1;  uint2 res = grp.count;
        for(uint offs = 1; offs < UNIT_WIDTH; offs *= 2)
        {
            buf[cur] = res;  barrier(CLK_LOCAL_MEM_FENCE);
            res += buf[cur - offs];  barrier(CLK_LOCAL_MEM_FENCE);
        }
        buf[cur] = res;  barrier(CLK_LOCAL_MEM_FENCE);
        if(!pos)
        {
            data->pixel_offset += data->pixel_count;  data->pixel_count = grp.count.s0;
        }
        grp.offset = offset + res - grp.count;  offset += buf[2 * UNIT_WIDTH - 1];
        grp_data[pos] = grp;  barrier(CLK_LOCAL_MEM_FENCE);
    }
    for(uint pos = index; pos < n; pos += UNIT_WIDTH)grp_data[pos].offset.s1 += offset.s0;
    if(index)return;  data->old_count = data->ray_count;  data->ray_count = offset.s0;
}

KERNEL void set_ray_index(const global GlobalData *data, const global GroupData *grp_data,
    const global uint2 *src_index, global uint2 *dst_index)
{
    uint index = get_global_id(0);  uint2 src = src_index[index];
    GroupData grp = grp_data[src.s0 & GROUP_ID_MASK];
    index -= (index < data->old_count ? grp.base.s0 : grp.base.s1);
    if(index < grp.count.s1)index += grp.offset.s1;
    else index += grp.offset.s0 - grp.count.s1;
    dst_index[index] = src;
}


KERNEL void update_image(global GlobalData *data, global float4 *area, write_only image2d_t image)
{
    uint index = get_global_id(0), width = data->cam.width;  float4 color = area[index];
    write_imagef(image, (int2)(index % width, index / width), pow(color / (color.w + 1e-6), 1 / 2.2));
}
