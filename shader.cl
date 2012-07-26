// shader.cl -- shader functions
//



uint sky_shader(global float4 *area, global RayQueue *ray, const global MatShader *shader)
{
    float3 color = (float3)(0.5, 1.0, 1.0);
    area[ray->pixel] += ray->weight * (float4)(color, 1);
    return ray->queue[0].group_id = spawn_group;
}

uint light_shader(global float4 *area, global RayQueue *ray, const global MatShader *shader)
{
    const float3 color = (float3)(1, 1, 1);
    area[ray->pixel] += ray->weight * (float4)(color, 1);
    return ray->queue[0].group_id = spawn_group;
}

uint mat_shader(global float4 *area, global RayQueue *ray, const global MatShader *shader)
{
    const float3 light = normalize((float3)(1, -1, 1));

    const float alpha = 100, f0 = 0.04;
    float3 dir = ray->ray.dir, norm = normalize(ray->norm), hvec = normalize(light - dir);
    float spec = (alpha + 2) / 8 * pow(max(0.0, dot(norm, hvec)), alpha);
    spec *= f0 + (1 - f0) * pow(max(0.0, -dot(dir, hvec)), 5);
    float3 color = (shader->color.xyz + spec * shader->color.w) * max(0.0, dot(light, norm));
    color += 0.1 * shader->color.xyz;  // ambient;

    //float4 weight = ray->weight;
    //area[ray->pixel] += 0.5 * weight * (float4)(color, 1);  ray->weight = 0.5 * weight;

    ray->weight *= (float4)(color, 1);  ray->type = rt_shadow;
    ray->ray.start_min.xyz += ray->ray.max * dir;  ray->ray.dir_max.xyz = light;
    return reset_ray(ray, ray->root.group_id, ray->root.local_id, light_group);

    /*ray->ray.start_min.xyz += ray->ray.max * dir;
    ray->ray.dir_max.xyz = dir - 2 * dot(dir, norm) * norm;
    return reset_ray(ray, ray->root.group_id, ray->root.local_id, sky_group);*/
}


#define GRID_SIZE           0x80
#define GRID_MAX    (2 * GRID_SIZE - 1)
#define GRID_SCALE          0.1f

float4 grid_transform(uint2 local_id)  // (x, y, rotate, scale)
{
    uint seed = calc_crc(calc_crc(calc_crc(local_id.s0 ^ local_id.s1) ^ local_id.s0) ^ local_id.s1);
    float2 pos = convert_float2((local_id.s0 >> (uint2)(0, 16)) & 0xFFFF) - GRID_SIZE;
    pos += convert_float2((seed >> (uint2)(0, 16)) & 0xFFFF) / 65536.0f;
    //pos += convert_float2(deinterleave(seed)) / 65536.0f;

    return (float4)(pos, calc_crc(seed) / 4294967296.0f, GRID_SCALE);
}

void transform(uint group_id, const global RayQueue *ray, Ray *res, float3 norm_mat[3], const global Matrix *mat_list)
{
    switch((group_id >> GROUP_TR_SHIFT) & GROUP_TR_MASK)
    {
    case tr_identity:
        norm_mat[0] = (float3)(1, 0, 0);
        norm_mat[1] = (float3)(0, 1, 0);
        norm_mat[2] = (float3)(0, 0, 1);
        *res = ray->ray;
        break;

    case tr_ortho:
        {
            Matrix mat = mat_list[ray->queue[0].local_id.s0];
            norm_mat[0] = (float3)(mat.x.x, mat.y.x, mat.z.x);
            norm_mat[1] = (float3)(mat.x.y, mat.y.y, mat.z.y);
            norm_mat[2] = (float3)(mat.x.z, mat.y.z, mat.z.z);

            *res = ray->ray;
            float3 rel = res->start - (float3)(mat.x.w, mat.y.w, mat.z.w);
            res->start_min.xyz = (mat.x * rel.x + mat.y * rel.y + mat.z * rel.z).xyz;
            res->dir_max.xyz = (mat.x * res->dir.x + mat.y * res->dir.y + mat.z * res->dir.z).xyz;
            break;
        }

    case tr_affine:
        // TODO
        break;

    case tr_grid:
        {
            float4 trans = grid_transform(ray->queue[0].local_id);
            float cos_a, sin_a = sincos(trans.z, &cos_a);
            norm_mat[0] = (float3)(cos_a, sin_a, 0);
            norm_mat[1] = (float3)(-sin_a, cos_a, 0);
            norm_mat[2] = (float3)(0, 0, 1);

            *res = ray->ray;
            float2 rel = res->start.xy - trans.xy;
            float2 sin_pm = (float2)(sin_a, -sin_a);
            res->start_min.xy = rel * cos_a + rel.yx * sin_pm;
            res->dir_max.xy = res->dir.xy * cos_a + res->dir.yx * sin_pm;
            res->start_min.xyz /= trans.w;  break;
        }
    }
}


uint grid_shader(const Ray *ray, const global GridShader *shader,
    const global RayHit *cur, RayHit *hit)
{
    float radius = shader->radius, height = shader->height;
    float3 max_coord = (float3)((float2)(GRID_SIZE + radius), height);
    float3 min_coord = (float3)(-max_coord.xy, 0);

    float3 inv_dir = 1 / ray->dir;
    float3 pos1 = (min_coord - ray->start) * inv_dir;
    float3 pos2 = (max_coord - ray->start) * inv_dir;
    float3 pos_min = min(pos1, pos2), pos_max = max(pos1, pos2);
    float t_min = max(max(pos_min.x, pos_min.y), max(pos_min.z, ray->min));
    float t_max = min(min(pos_max.x, pos_max.y), min(pos_max.z, ray->max));
    if(!(t_max > t_min))return 0;

    uint2 cell = convert_uint2_rtn(ray->start.xy +
        t_min * ray->dir.xy + sign(ray->dir.xy) * radius + GRID_SIZE);
    hit[0].pos = t_min;  hit[0].group_id = shader->cell_group;
    hit[0].local_id.s0 = cell.s0 | cell.s1 << 16;
    hit[0].local_id.s1 = 0;  return 1;
}

uint cell_shader(const Ray *ray, const global GridShader *shader,
    const global RayHit *cur, RayHit *hit)
{
    uint cell_id = cur->local_id.s0;
    uint2 cell = (cell_id >> (uint2)(0, 16)) & 0xFFFF;
    float2 pos = convert_float2(cell) - GRID_SIZE;

    float radius = shader->radius, height = shader->height;
    float3 coord = (float3)(pos + 0.5f, height / 2) +
        sign(ray->dir) * (float3)((float2)(0.5f - radius), height / 2);
    float3 t = (coord - ray->start) / ray->dir;

    uint hit_count = 0;
    float t_next = min(t.x, t.y);
    if(t_next < t.z)
    {
        uint index = t.x > t.y;  uint2 perm = (uint2)(index, 1 - index);
        int2 delta = select((int2)1, (int2)(-1), shuffle(ray->dir.xy, perm) < 0);

        int2 old_pos = shuffle(convert_int2(cell), perm);
        int2 new_pos = (int2)(min(max(old_pos.s0 + delta.s0, 0), GRID_MAX), old_pos.s1);
        if(new_pos.s0 != old_pos.s0)
        {
            float2 offs = ray->start.xy + t_next * ray->dir.xy - coord.xy;
            if(fabs(fabs(shuffle(offs, perm).s1) - (1 - radius)) < radius)
                new_pos.s1 = min(max(old_pos.s1 - delta.s1, 0), GRID_MAX);
            new_pos = shuffle(new_pos, perm);

            hit[0].pos = t_next;  hit[0].group_id = shader->cell_group;
            hit[0].local_id.s0 = (uint)new_pos.s0 | (uint)new_pos.s1 << 16;
            hit[0].local_id.s1 = 0;  hit_count = 1;
        }
    }

    const uint n = 32;
    uint obj_group = shader->obj_group;
    const float inv_scale = 1 / GRID_SCALE;
    for(uint i = 0; i < n; i++)
    {
        float4 trans = grid_transform((uint2)(cell_id, i));
        trans.w *= inv_scale;

        float R2 = radius * trans.w;  R2 *= R2;
        float h = height * trans.w;

        float2 offs = trans.xy - ray->start.xy;
        float pos = dot(offs, ray->dir.xy) / dot(ray->dir.xy, ray->dir.xy);
        offs -= pos * ray->dir.xy;  float r2 = dot(offs, offs);  if(!(r2 < R2))continue;

        float2 bound = pos + sqrt(R2 - r2) * (float2)(-1, 1);
        if(!(bound.s0 < ray->max && bound.s1 > ray->min))continue;
        float2 z = ray->start.z + bound * ray->dir.z;
        if(!(min(z.s0, z.s1) < h && max(z.s0, z.s1) > 0))continue;

        if(hit_count >= MAX_HITS)return MAX_HITS;  // artifacts

        hit[hit_count].pos = bound.s0;
        hit[hit_count].group_id = obj_group;
        hit[hit_count].local_id = (uint2)(cell_id, i);
        hit_count++;
    }
    return hit_count;
}


uint aabb_shader(const Ray *ray, const global AABBShader *shader,
    const global RayHit *cur, RayHit *hit, const global AABB *aabb)
{
    aabb += shader->aabb_offs;
    float3 inv_dir = 1 / ray->dir;
    uint n = shader->aabb_count, hit_count = 0;
    int2 flags = (shader->flags & (uint2)(f_local0, f_local1)) != 0;
    uint2 cur_local = cur->local_id;
    for(uint i = 0; i < n; i++)
    {
        float3 pos1 = (aabb[i].min - ray->start) * inv_dir;
        float3 pos2 = (aabb[i].max - ray->start) * inv_dir;
        float3 pos_min = min(pos1, pos2), pos_max = max(pos1, pos2);
        float t_min = max(max(pos_min.x, pos_min.y), max(pos_min.z, ray->min));
        float t_max = min(min(pos_max.x, pos_max.y), min(pos_max.z, ray->max));
        if(!(t_max > t_min))continue;

        if(hit_count >= MAX_HITS)return MAX_HITS;  // artifacts

        hit[hit_count].pos = t_min;
        hit[hit_count].group_id = aabb[i].group_id;
        hit[hit_count].local_id = select(cur_local, aabb[i].local_id, flags);
        hit_count++;
    }
    return hit_count;
}


uint sphere_shader(const Ray *ray, uint material_id, float4 *norm_pos)
{
    const float R2 = 1;
    const float3 center = (float3)(0, 0, 0);

    float3 offs = center - ray->start;  float pos = dot(offs, ray->dir);
    offs -= pos * ray->dir;  float r2 = dot(offs, offs);  if(r2 > R2)return 0xFFFFFFFF;
    pos -= sqrt(R2 - r2);  if(!(pos >= ray->min && pos < ray->max))return 0xFFFFFFFF;
    *norm_pos = (float4)(ray->start + pos * ray->dir, pos);  return material_id;
}

uint mesh_shader(const Ray *ray, const global MeshShader *shader,
    float4 *norm_pos, const global Vertex *vtx, const global uint *tri)
{
    //*norm_pos = (float4)(-ray->dir, ray->min);  return shader->material_id;  // DEBUG

    //return sphere_shader(ray, shader->material_id, norm_pos);

    vtx += shader->vtx_offs;  tri += shader->tri_offs;
    uint hit_index = 0xFFFFFFFF, n = shader->tri_count;
    float hit_u, hit_v;  norm_pos->w = ray->max;
    for(uint i = 0; i < n; i++)
    {
        uint3 index = (tri[i] >> (uint3)(0, 10, 20)) & 0x3FF;
        float3 r = vtx[index.s0].pos, p = vtx[index.s1].pos - r, q = vtx[index.s2].pos - r;  r -= ray->start;
        float3 n = cross(p, q);  float w = dot(ray->dir, n);  /*if(!(w > 0))continue;*/  w = 1 / w;
        float t = dot(r, n) * w;  if(!(t >= ray->min && t < norm_pos->w))continue;
        float3 dr = cross(ray->dir, r);  float u = -dot(q, dr) * w, v = dot(p, dr) * w;
        if(!(u >= 0 && v >= 0 && u + v <= 1))continue;

        norm_pos->w = t;  hit_u = u;  hit_v = v;  hit_index = i;
    }
    if(hit_index == 0xFFFFFFFF)return 0xFFFFFFFF;

    uint3 index = (tri[hit_index] >> (uint3)(0, 10, 20)) & 0x3FF;
    norm_pos->xyz = vtx[index.s0].norm * (1 - hit_u - hit_v) + vtx[index.s1].norm * hit_u + vtx[index.s2].norm * hit_v;
    return shader->material_id;
}
