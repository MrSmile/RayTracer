// shader.cl -- shader functions
//


uint sky_shader(global GlobalData *data, global float4 *area, global RayUnit *ray)
{
    const uint index = get_local_id(0), pixel = ray->pixel[index];
    float3 color = 0.5 + 0.5 * (float3)(ray->dir_x[index], ray->dir_y[index], ray->dir_z[index]);
    area[pixel].s0 += ray->weight_r[index] * color.s0;
    area[pixel].s1 += ray->weight_g[index] * color.s1;
    area[pixel].s2 += ray->weight_b[index] * color.s2;
    area[pixel].s3 += ray->weight_w[index];
    return ray->queue[0].group_id[index] = spawn_group;
}

uint mat_shader(global GlobalData *data, global float4 *area, global RayUnit *ray)
{
    const uint index = get_local_id(0);
    float3 norm = normalize((float3)(ray->norm_x[index], ray->norm_y[index], ray->norm_z[index]));
    const float3 light = normalize((float3)(1, -1, 1));  float3 color = max(0.0, dot(light, norm));
    float4 weight = (float4)(ray->weight_r[index], ray->weight_g[index], ray->weight_b[index], ray->weight_w[index]);
    //area[ray->pixel[index]] += weight * (float4)(color, 1);
    //return ray->queue[0].group_id[index] = spawn_group;

    area[ray->pixel[index]] += 0.5 * weight * (float4)(color, 1);  weight *= 0.5;
    ray->weight_r[index] = weight.s0;  ray->weight_g[index] = weight.s1;
    ray->weight_b[index] = weight.s2;  ray->weight_w[index] = weight.s3;

    float3 start = (float3)(ray->start_x[index], ray->start_y[index], ray->start_z[index]);
    float3 dir = (float3)(ray->dir_x[index], ray->dir_y[index], ray->dir_z[index]);
    float3 hit = start + ray->max[index] * dir, reflect = dir - 2 * dot(dir, norm) * norm;
    ray->start_x[index] = hit.x;  ray->start_y[index] = hit.y;  ray->start_z[index] = hit.z;
    ray->dir_x[index] = reflect.x;  ray->dir_y[index] = reflect.y;  ray->dir_z[index] = reflect.z;
    return reset_ray(ray, ray->root_group[index], ray->root_local0[index], ray->root_local1[index]);
}

KERNEL void update_image(global GlobalData *data, global float4 *area, write_only image2d_t image)
{
    uint index = get_global_id(0), width = data->cam.width;  float4 color = area[index];
    write_imagef(image, (int2)(index % width, index / width), color / (color.w + 1e-6));
}


void transform(uint group_id, const global RayUnit *ray, Ray *res, float3 res_mat[4], const global Matrix *mat_list)
{
    const uint index = get_local_id(0);
    switch((group_id >> GROUP_TR_SHIFT) & GROUP_TR_MASK)
    {
    case tr_identity:
        res_mat[0] = (float3)(1, 0, 0);
        res_mat[1] = (float3)(0, 1, 0);
        res_mat[2] = (float3)(0, 0, 1);
        res_mat[3] = (float3)(0, 0, 0);
        res->start_min = (float4)(ray->start_x[index], ray->start_y[index], ray->start_z[index], ray->min[index]);
        res->dir_max = (float4)(ray->dir_x[index], ray->dir_y[index], ray->dir_z[index], ray->max[index]);
        break;

    case tr_ortho:
        {
            Matrix mat = mat_list[ray->queue[0].local0_id[index]];
            res_mat[0] = (float3)(mat.x.x, mat.y.x, mat.z.x);
            res_mat[1] = (float3)(mat.x.y, mat.y.y, mat.z.y);
            res_mat[2] = (float3)(mat.x.z, mat.y.z, mat.z.z);
            res_mat[3] = (float3)(mat.x.w, mat.y.w, mat.z.w);

            res->start_min = (float4)(ray->start_x[index], ray->start_y[index], ray->start_z[index], ray->min[index]);
            res->dir_max = (float4)(ray->dir_x[index], ray->dir_y[index], ray->dir_z[index], ray->max[index]);

            float3 rel = res->start - res_mat[3];
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


bool sphere_shader(const Ray *ray, uint material_id, global RayHitUnit *cur, RayStop *stop)
{
    const float R2 = 1;
    const float3 center = (float3)(0, 0, 0);

    const uint index = get_local_id(0);
    float3 offs = center - ray->start;  float pos = dot(offs, ray->dir);
    offs -= pos * ray->dir;  float r2 = dot(offs, offs);  if(r2 > R2)return false;
    pos -= sqrt(R2 - r2);  if(!(pos > ray->min && pos < ray->max))return false;

    stop->norm = ray->start + pos * ray->dir;
    RayHit orig = {pos, cur->group_id[index], (uint2)(cur->local0_id[index], cur->local1_id[index])};
    stop->orig = orig;  stop->material_id = material_id;  return true;
}

bool mesh_shader(const Ray *ray, const global MeshShader *shader, global RayHitUnit *cur,
    RayStop *stop, const global Vertex *vtx, const global uint *tri)
{
    //return sphere_shader(ray, shader->material_id, cur, stop);

    const uint index = get_local_id(0);
    vtx += shader->vtx_offs;  tri += shader->tri_offs;
    uint hit_index = 0xFFFFFFFF;  float hit_u, hit_v, pos = ray->max;
    for(uint i = 0; i < shader->tri_count; i++)
    {
        uint3 pt = (tri[i] >> (uint3)(0, 10, 20)) & 0x3FF;
        float3 r = vtx[pt.s0].pos, p = vtx[pt.s1].pos - r, q = vtx[pt.s2].pos - r;  r -= ray->start;
        float3 n = cross(p, q);  float w = dot(ray->dir, n);  if(!(w > 0))continue;  w = 1 / w;
        float t = dot(r, n) * w;  if(!(t > ray->min && t < pos))continue;
        float3 dr = cross(ray->dir, r);  float u = -dot(q, dr) * w, v = dot(p, dr) * w;
        if(!(u >= 0 && v >= 0 && u + v <= 1))continue;

        pos = t;  hit_u = u;  hit_v = v;  hit_index = i;
    }
    if(hit_index == 0xFFFFFFFF)return false;

    uint3 pt = (tri[hit_index] >> (uint3)(0, 10, 20)) & 0x3FF;
    stop->norm = vtx[pt.s0].norm * (1 - hit_u - hit_v) + vtx[pt.s1].norm * hit_u + vtx[pt.s2].norm * hit_v;
    RayHit orig = {pos, cur->group_id[index], (uint2)(cur->local0_id[index], cur->local1_id[index])};
    stop->orig = orig;  stop->material_id = shader->material_id;  return true;
}
