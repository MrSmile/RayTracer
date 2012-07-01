// aabb-list.cl -- calculations of AABB-ray intersections
//


uint process_aabb_list(Ray *ray, AABBGroup *grp, RayHit *hit, global AABB *aabb)
{
    uint hit_count = 0;
    float3 inv_dir = 1 / ray->dir;
    global AABB *aabb = grp->aabb;
    for(uint i = 0; i < grp->count; i++)
    {
        float3 pos1 = (aabb[i].min - ray->start) * inv_dir;
        float3 pos2 = (aabb[i].max - ray->start) * inv_dir;
        float3 pos_min = min(pos1, pos2), pos_max = max(pos1, pos2);
        float t_min = min(min(pos_min.x, pos_min.y), pos_min.z);
        float t_max = min(min(pos_max.x, pos_max.y), pos_max.z);
        if(!(t_max > ray->min && t_min < ray->max))continue;

        hit[hit_count].pos = t_min;  hit[hit_count].id_group = aabb[i].id_group;
        hit[hit_count].id_local = (uint2)(aabb[i].id_local, 0);  hit_count++;
    }
    return hit_count;
}
