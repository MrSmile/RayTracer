// shader.cl -- shader functions
//



void transform(Ray *ray, uint transform_id, const Transform *trans, float3 mat[4])
{
    switch(transform_id)
    {
        case tr_identity:
            mat[0] = (float3)(1, 0, 0);
            mat[1] = (float3)(0, 1, 0);
            mat[2] = (float3)(0, 0, 1);
            break;

        case tr_matrix:
            // TODO
            break;
    }
}


uint process_aabb_list(Ray *ray, const AABBShader *shader, RayHit *hit, global const AABB *aabb)
{
    float3 inv_dir = 1 / ray->dir;
    aabb += shader->aabb_offs;  uint hit_count = 0;
    for(uint i = 0; i < shader->count; i++)
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


bool process_tri_list(Ray *ray, const TriShader *shader, RayHit cur,
    RayStop *stop, global const Vertex *vtx, global const uint *tri)
{
    uint hit_index = 0xFFFFFFFF;  float hit_w;
    vtx += shader->vtx_offs;  tri += shader->tri_offs;
    for(uint i = 0; i < shader->tri_count; i++)
    {
        uint3 index = (tri[i] >> (uint3)(0, 10, 20)) & 0x3FF;
        float3 r = vtx[index.s0].pos, p = vtx[index.s1].pos - r, q = vtx[index.s2].pos - r;  r -= ray->start;
        float3 n = cross(p, q);  float w = 1 / dot(ray->dir, n), t = dot(r, n) * w;  // w sign -- cull mode
        if(!(t > ray->min && t < ray->max))continue;  cur.pos = t;  hit_w = w;  hit_index = i;
    }
    if(hit_index == 0xFFFFFFFF)return false;

    uint3 index = (tri[hit_index] >> (uint3)(0, 10, 20)) & 0x3FF;
    float3 r = vtx[index.s0].pos, p = vtx[index.s1].pos - r, q = vtx[index.s2].pos - r;  r -= ray->start;
    float3 dr = cross(ray->dir, r);  float u = -dot(q, dr) * hit_w, v = dot(p, dr) * hit_w;
    stop->norm = vtx[index.s0].norm * (1 - u - v) + vtx[index.s1].norm * u + vtx[index.s2].norm * v;
    stop->orig = cur;  stop->material_id = shader->material_id;  return true;
}
