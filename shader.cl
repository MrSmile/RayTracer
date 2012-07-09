// shader.cl -- shader functions
//



uint sky_shader(global GlobalData *data, global float4 *area, global RayQueue *ray)
{
    float3 color = 0.5 + 0.5 * ray->ray.dir;
    area[ray->pixel] += ray->weight * (float4)(color, 1);
    return ray->queue[0].group_id = spawn_group;
}

uint mat_shader(global GlobalData *data, global float4 *area, global RayQueue *ray)
{
    float3 norm = normalize(ray->stop.norm);
    const float3 light = normalize((float3)(1, -1, 1));
    float3 color = max(0.0, dot(light, norm));  float4 weight = ray->weight;
    area[ray->pixel] += 0.5 * weight * (float4)(color, 1);  ray->weight = 0.5 * weight;

    float3 dir = ray->ray.dir;
    ray->ray.start_min.xyz += ray->ray.max * dir;
    ray->ray.dir_max.xyz = dir - 2 * dot(dir, norm) * norm;
    return reset_ray(ray, ray->root.group_id, ray->root.local_id);
}

KERNEL void update_image(global GlobalData *data, global float4 *area, write_only image2d_t image)
{
    uint index = get_global_id(0), width = data->cam.width;  float4 color = area[index];
    write_imagef(image, (int2)(index % width, index / width), color / (color.w + 1e-6));
}


void transform(uint group_id, const global RayQueue *ray, Ray *res, float3 res_mat[4], const global Matrix *mat_list) 
{
    switch((group_id >> GROUP_TR_SHIFT) & GROUP_TR_MASK)
    {
    case tr_identity:
        res_mat[0] = (float3)(1, 0, 0);
        res_mat[1] = (float3)(0, 1, 0);
        res_mat[2] = (float3)(0, 0, 1);
        res_mat[3] = (float3)(0, 0, 0);
        *res = ray->ray;
        break;

    case tr_ortho:
        {
            Matrix mat = mat_list[ray->queue[0].local_id.s0];
            res_mat[0] = (float3)(mat.x.x, mat.y.x, mat.z.x);
            res_mat[1] = (float3)(mat.x.y, mat.y.y, mat.z.y);
            res_mat[2] = (float3)(mat.x.z, mat.y.z, mat.z.z);
            res_mat[3] = (float3)(mat.x.w, mat.y.w, mat.z.w);

            *res = ray->ray;  float3 rel = res->start - res_mat[3];
            res->start_min.xyz = (mat.x * rel.x + mat.y * rel.y + mat.z * rel.z).xyz;
            res->dir_max.xyz = (mat.x * res->dir.x + mat.y * res->dir.y + mat.z * res->dir.z).xyz;
            break;
        }

    case tr_affine:
        // TODO
        break;
    }
}


uint aabb_shader(const Ray *ray, const global AABBShader *shader, RayHit *hit, const global AABB *aabb)
{
    aabb += shader->aabb_offs;
    float3 inv_dir = 1 / ray->dir;
    uint n = shader->count, hit_count = 0;
    for(uint i = 0; i < n; i++)
    {
        float3 pos1 = (aabb[i].min - ray->start) * inv_dir;
        float3 pos2 = (aabb[i].max - ray->start) * inv_dir;
        float3 pos_min = min(pos1, pos2), pos_max = max(pos1, pos2);
        float t_min = min(min(pos_min.x, pos_min.y), pos_min.z);
        float t_max = min(min(pos_max.x, pos_max.y), pos_max.z);
        if(!(t_max > ray->min && t_min < ray->max))continue;

        hit[hit_count].pos = t_min;  hit[hit_count].group_id = aabb[i].group_id;
        hit[hit_count].local_id = (uint2)(aabb[i].local_id, 0);  hit_count++;
    }
    return hit_count;
}


bool sphere_shader(const Ray *ray, uint material_id, global RayHit *cur, RayStop *stop)
{
    const float R2 = 1;
    const float3 center = (float3)(0, 0, 0);

    float3 offs = center - ray->start;  float pos = dot(offs, ray->dir);
    offs -= pos * ray->dir;  float r2 = dot(offs, offs);  if(r2 > R2)return false;
    pos -= sqrt(R2 - r2);  if(!(pos > ray->min && pos < ray->max))return false;

    stop->norm = ray->start + pos * ray->dir;
    RayHit orig = {pos, cur->group_id, cur->local_id};  stop->orig = orig;
    stop->material_id = material_id;  return true;
}

bool mesh_shader(const Ray *ray, const global MeshShader *shader, global RayHit *cur,
    RayStop *stop, const global Vertex *vtx, const global uint *tri)
{
    //return sphere_shader(ray, shader->material_id, cur, stop);

    vtx += shader->vtx_offs;  tri += shader->tri_offs;
    uint hit_index = 0xFFFFFFFF, n = shader->tri_count;
    float hit_u, hit_v, pos = ray->max;
    for(uint i = 0; i < n; i++)
    {
        uint3 index = (tri[i] >> (uint3)(0, 10, 20)) & 0x3FF;
        float3 r = vtx[index.s0].pos, p = vtx[index.s1].pos - r, q = vtx[index.s2].pos - r;  r -= ray->start;
        float3 n = cross(p, q);  float w = dot(ray->dir, n);  if(!(w > 0))continue;  w = 1 / w;
        float t = dot(r, n) * w;  if(!(t > ray->min && t < pos))continue;
        float3 dr = cross(ray->dir, r);  float u = -dot(q, dr) * w, v = dot(p, dr) * w;
        if(!(u >= 0 && v >= 0 && u + v <= 1))continue;

        pos = t;  hit_u = u;  hit_v = v;  hit_index = i;
    }
    if(hit_index == 0xFFFFFFFF)return false;

    uint3 index = (tri[hit_index] >> (uint3)(0, 10, 20)) & 0x3FF;
    stop->norm = vtx[index.s0].norm * (1 - hit_u - hit_v) + vtx[index.s1].norm * hit_u + vtx[index.s2].norm * hit_v;
    RayHit orig = {pos, cur->group_id, cur->local_id};  stop->orig = orig;
    stop->material_id = shader->material_id;  return true;
}
