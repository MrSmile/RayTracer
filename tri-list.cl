// tri-list.cl -- calculations of triangle-ray intersections
//


bool process_tri_list(Ray *ray, TriGroup *grp, RayStop *stop, global Vertex *vtx, global uint *tri)
{
    uint hit_index = 0xFFFFFFFF;  float hit_w;
    for(uint i = 0; i < grp->tri_count; i++)
    {
        uint3 index = (tri[i] >> (uint3)(0, 10, 20)) & 0x3FF;
        float3 r = vtx[index.s0].pos, p = vtx[index.s1].pos - r, q = vtx[index.s2].pos - r;  r -= ray->start;
        float3 n = cross(p, q);  float w = 1 / dot(ray->dir, n), t = dot(r, n) * w;  // w sign -- cull mode
        if(!(t > ray->min && t < ray->max))continue;  stop->hit.pos = t;  hit_w = w;  hit_index = i;
    }
    if(hit_index == 0xFFFFFFFF)return false;

    uint3 index = (tri[hit_index] >> (uint3)(0, 10, 20)) & 0x3FF;
    float3 r = vtx[index.s0].pos, p = vtx[index.s1].pos - r, q = vtx[index.s2].pos - r;  r -= ray->start;
    float3 dr = cross(ray->dir, r);  float u = -dot(q, dr) * hit_w, v = dot(p, dr) * hit_w;
    stop->norm = vtx[index.s0].norm * (1 - u - v) + vtx[index.s1].norm * u + vtx[index.s2].norm * v;
    stop->hit.id_group = grp->id_group;  stop->hit.id_local = grp->id_local;  return true;
}
