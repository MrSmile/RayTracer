// ray-tracer.cl -- main kernel
//

#define KERNEL  kernel __attribute__((reqd_work_group_size(UNIT_WIDTH, 1, 1)))

#include "ray-tracer.h"


uint reset_ray(global RayUnit *ray, uint group_id, uint local0_id, uint local1_id)
{
    const uint index = get_local_id(0);
    ray->orig_group[index] = ray->queue[0].group_id[index] = group_id;
    ray->orig_local0[index] = ray->queue[0].local0_id[index] = local0_id;
    ray->orig_local1[index] = ray->queue[0].local1_id[index] = local1_id;
    ray->queue[0].pos[index] = ray->min[index] = 0;  ray->max[index] = INFINITY;
    ray->queue_len[index] = 1;  ray->material_id[index] = sky_group;  return group_id;
}


#include "shader.cl"


KERNEL void init_groups(global GroupData *grp_data)
{
    grp_data[get_global_id(0)].cur_index = 0;
}

uint init_ray(const global Camera *cam, global RayUnit *ray, uint pixel)
{
    const uint index = get_local_id(0);
    pixel %= cam->width * cam->height;  // TODO: shuffle
    float x = pixel % cam->width + 0.5, y = pixel / cam->width + 0.5;  // TODO: randomize
    ray->pixel[index] = pixel;

    ray->weight_r[index] = ray->weight_g[index] = ray->weight_b[index] = ray->weight_w[index] = 1;

    float3 dir = normalize(cam->top_left + x * cam->dx + y * cam->dy);
    ray->start_x[index] = cam->eye.x;  ray->start_y[index] = cam->eye.y;  ray->start_z[index] = cam->eye.z;
    ray->dir_x[index] = dir.x;  ray->dir_y[index] = dir.y;  ray->dir_z[index] = dir.z;
    return reset_ray(ray, ray->root_group[index] = cam->root_group,
        ray->root_local0[index] = cam->root_local, ray->root_local1[index] = 0);
}

KERNEL void init_rays(global GlobalData *data, global RayUnit *ray)
{
    const uint pixel = get_global_id(0);
    init_ray(&data->cam, &ray[get_group_id(0)], pixel);  if(pixel)return;
    data->pixel_offset = get_global_size(0);  data->pixel_count = 0;
}

KERNEL void init_image(global float4 *area)
{
    area[get_global_id(0)] = 0;
}


void heap_sort(RayHit *hit, uint n, uint max)  // TODO: rewrite without branches
{
    RayHit cur = hit[n];  uint k = n;
    for(; k > 1 && hit[k / 2].pos < cur.pos; k /= 2)hit[k] = hit[k / 2];
    hit[k] = cur;

    //if(n < max)heap_sort(hit, n + 1, max);  // compiler segfault here

    cur = hit[n];  hit[n] = hit[1];
    for(k = 2; k <= n; k *= 2)
    {
        if(k < n && hit[k + 1].pos > hit[k].pos)k++;
        if(hit[k].pos > cur.pos)hit[k / 2] = hit[k];
        else break;
    }
    hit[k / 2] = cur;
}

void sort_hits(RayHit *hit, uint n)
{
    if(n > 1)heap_sort(hit - 1, 2, n);  hit[n].pos = INFINITY;
}

KERNEL void process(global GlobalData *data, global float4 *area,
    global GroupData *grp_data, global RayUnit *ray,
    const global Group *grp_list, const global Matrix *mat_list,
    const global AABB *aabb, const global Vertex *vtx, const global uint *tri)
{
    if(get_global_id(0) >= data->ray_count)return;
    const uint index = get_local_id(0);  ray += get_group_id(0);
    uint group_id = ray->queue[0].group_id[index];

    Ray cur;  float3 mat[4];  uint queue_len, n;
    transform(group_id, ray, &cur, mat, mat_list);
    RayHit hit[MAX_QUEUE_LEN + MAX_HITS + 1];  RayStop stop;
    switch((group_id >> GROUP_SH_SHIFT) & GROUP_SH_MASK)
    {
    case sh_spawn:
        group_id = init_ray(&data->cam, ray, get_global_id(0) + data->pixel_offset);  goto assign_index;

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

    queue_len = ray->queue_len[index] - 1;
    for(uint i = 0; i < queue_len; i++)
    {
        hit[i].pos = ray->queue[i + 1].pos[index];
        hit[i].group_id = ray->queue[i + 1].group_id[index];
        hit[i].local_id.s0 = ray->queue[i + 1].local0_id[index];
        hit[i].local_id.s1 = ray->queue[i + 1].local1_id[index];
    }

save_queue:
    if(queue_len)
    {
        ray->queue_len[index] = queue_len;  ray->min[index] = ray->queue[0].pos[index];
    }
    else
    {
        hit[0].group_id = ray->material_id[index];  hit[0].local_id = 0;  queue_len = 1;
    }

copy_queue:
    for(uint i = 0; i < queue_len; i++)
    {
        ray->queue[i].pos[index] = hit[i].pos;
        ray->queue[i].group_id[index] = hit[i].group_id;
        ray->queue[i].local0_id[index] = hit[i].local_id.s0;
        ray->queue[i].local1_id[index] = hit[i].local_id.s1;
    }
    group_id = hit[0].group_id;

assign_index:
    ray->index[index] = atomic_add(&grp_data[group_id & GROUP_ID_MASK].cur_index, 1);  // TODO: optimize
    return;

insert_stop:
    queue_len = ray->queue_len[index] - 1;
    for(uint i = 0; i < queue_len; i++)
    {
        hit[i].pos = ray->queue[i + 1].pos[index];
        if(hit[i].pos >= stop.orig.pos)
        {
            queue_len = i;  break;
        }
        hit[i].group_id = ray->queue[i + 1].group_id[index];
        hit[i].local_id.s0 = ray->queue[i + 1].local0_id[index];
        hit[i].local_id.s1 = ray->queue[i + 1].local1_id[index];
    }
    if(queue_len)
    {
        ray->queue_len[index] = queue_len;  ray->min[index] = ray->queue[0].pos[index];
    }
    else
    {
        hit[0].group_id = stop.material_id;  hit[0].local_id = 0;  queue_len = 1;
    }
    ray->max[index] = stop.orig.pos;  ray->orig_group[index] = stop.orig.group_id;
    ray->orig_local0[index] = stop.orig.local_id.s0;  ray->orig_local1[index] = stop.orig.local_id.s1;
    stop.norm = mat[0] * stop.norm.x + mat[1] * stop.norm.y + mat[2] * stop.norm.z;
    ray->norm_x[index] = stop.norm.x;  ray->norm_y[index] = stop.norm.y;  ray->norm_z[index] = stop.norm.z;
    ray->material_id[index] = stop.material_id;  goto copy_queue;

insert_hits:
    sort_hits(hit + MAX_QUEUE_LEN, n);
    uint old_len = ray->queue_len[index];  queue_len = 0;
    for(uint i = 1, index = MAX_QUEUE_LEN; i < old_len; i++)
    {
        float pos = ray->queue[i].pos[index];
        while(hit[index].pos < pos)
        {
            hit[queue_len++] = hit[index++];
            if(queue_len == MAX_QUEUE_LEN)goto overflow;
        }
        hit[queue_len].pos = pos;
        if(++queue_len == MAX_QUEUE_LEN)goto overflow;
        hit[queue_len - 1].group_id = ray->queue[i].group_id[index];
        hit[queue_len - 1].local_id.s0 = ray->queue[i].local0_id[index];
        hit[queue_len - 1].local_id.s1 = ray->queue[i].local1_id[index];
    }
    goto save_queue;

overflow:
    hit[MAX_QUEUE_LEN - 1].group_id = ray->root_group[index];
    hit[MAX_QUEUE_LEN - 1].local_id.s0 = ray->root_local0[index];
    hit[MAX_QUEUE_LEN - 1].local_id.s1 = ray->root_local1[index];
    goto save_queue;
}


KERNEL void update_groups(global GlobalData *data, global GroupData *grp_data)  // single unit
{
    local uint2 buf[2 * UNIT_WIDTH], *ptr = buf + UNIT_WIDTH;
    const uint index = get_global_id(0), n = data->group_count;  uint2 offs = 0;
    for(uint pos = index; pos < n; pos += UNIT_WIDTH)
    {
        GroupData grp;  grp.count.s0 = grp_data[pos].cur_index;
        grp.count.s1 = grp.count.s0 % UNIT_WIDTH;  grp.count.s0 -= grp.count.s1;
        buf[index] = 0;  uint cur = UNIT_WIDTH + index;  uint2 res = grp.count;
        for(uint offs = 1; offs < UNIT_WIDTH; offs *= 2)
        {
            buf[cur] = res;  barrier(CLK_LOCAL_MEM_FENCE);
            res += buf[cur - offs];  barrier(CLK_LOCAL_MEM_FENCE);
        }
        buf[cur] = res;  barrier(CLK_LOCAL_MEM_FENCE);

        grp.offset = offs + res - grp.count;  offs += buf[2 * UNIT_WIDTH - 1];
        grp_data[pos] = grp;  if(pos)continue;

        data->pixel_offset += data->pixel_count;  data->pixel_count = grp.count.s0;
    }
    for(uint pos = index; pos < n; pos += UNIT_WIDTH)grp_data[pos].offset.s1 += offs.s0;
    if(!index)data->ray_count = offs.s0;
}

KERNEL void shuffle_rays(const global GroupData *grp_data, const global RayUnitData *ray, global uint *ray_buf)
{
    RaySingleData src;  const uint index = get_local_id(0);  ray += get_group_id(0);
    for(uint i = 0; i < RAY_UNIT_HEIGHT; i++)src.data[i] = ray->data[i][index];

    local uint pos[UNIT_WIDTH];
    GroupData data = grp_data[src.ray.queue[0].group_id & GROUP_ID_MASK];
    if(src.ray.index < data.count.s0)pos[index] = data.offset.s0;
    else
    {
        src.ray.index -= data.count.s0;  pos[index] = data.offset.s1;
    }
    pos[index] += src.ray.index;

    local uint buf[UNIT_WIDTH];  uint dst[RAY_UNIT_HEIGHT];
    uint offs = index / W_TO_H, rem = index % W_TO_H;
    for(uint i = 0; i < RAY_UNIT_HEIGHT; i++)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        buf[(rem * RAY_UNIT_HEIGHT + offs) % UNIT_WIDTH] = src.data[offs];
        barrier(CLK_LOCAL_MEM_FENCE);  offs = (offs + 1) % RAY_UNIT_HEIGHT;
        dst[(index + RAY_UNIT_HEIGHT - i) % RAY_UNIT_HEIGHT] = buf[index];
    }
    offs = index / RAY_UNIT_HEIGHT;  rem = index % RAY_UNIT_HEIGHT;
    for(uint i = 0; i < RAY_UNIT_HEIGHT; i++)
    {
        uint target = pos[i * W_TO_H + offs];
        uint block_index = target / UNIT_WIDTH, block_offset = target % UNIT_WIDTH;
        ray_buf[block_index * BLOCK_SIZE + block_offset * RAY_UNIT_HEIGHT + rem] = dst[i];
    }
}

KERNEL void transpose_rays(const global uint *ray_buf, global RayUnitData *ray)
{
    uint src[RAY_UNIT_HEIGHT];  const uint index = get_local_id(0);
    ray += get_group_id(0);  ray_buf += BLOCK_SIZE * get_group_id(0);
    for(uint i = 0; i < RAY_UNIT_HEIGHT; i++)src[i] = ray_buf[i * UNIT_WIDTH + index];

    local uint buf[UNIT_WIDTH];  RaySingleData dst;
    uint offs = index / W_TO_H, rem = index % W_TO_H;
    for(uint i = 0; i < RAY_UNIT_HEIGHT; i++)
    {
        buf[index] = src[(index + RAY_UNIT_HEIGHT - i) % RAY_UNIT_HEIGHT];
        barrier(CLK_LOCAL_MEM_FENCE);
        dst.data[offs] = buf[(rem * RAY_UNIT_HEIGHT + offs) % UNIT_WIDTH];
        barrier(CLK_LOCAL_MEM_FENCE);  offs = (offs + 1) % RAY_UNIT_HEIGHT;
    }
    for(uint i = 0; i < RAY_UNIT_HEIGHT; i++)ray->data[i][index] = dst.data[i];
}
